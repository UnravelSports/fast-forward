"""End-to-end Ray `map_batches` integration test for engine='arrow[spark]'.

Mirrors tests/test_pyspark_udf.py for the Ray framework. Uses a local Ray
cluster (``ray.init(num_cpus=2)``); no real cluster required.

Ray 2.5x's internal pyarrow operations don't support Arrow ``string_view``
(``ArrowTypeError: Extracting byte ranges not supported for type
string_view``), so Ray needs the pre-normalized ``arrow[spark]`` engine
just like Spark does.

Gated on ray + pyarrow being installed.
"""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")
ray_data = pytest.importorskip("ray.data")
pa = pytest.importorskip("pyarrow")

from fastforward import skillcorner
from tests.config import SC_RAW, SC_META


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw_bytes():
    with open(SC_RAW, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_bytes():
    with open(SC_META, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def baseline_height(raw_bytes, meta_bytes):
    ds = skillcorner.load_tracking(
        raw_bytes, meta_bytes,
        engine="polars",
        include_ball_owning_player=True,
        include_is_detected=True,
    )
    return ds.tracking.height


@pytest.fixture(scope="module", autouse=True)
def ray_local():
    """Local Ray cluster for the module. ``num_cpus=2`` gives true
    parallelism without overcommitting.
    """
    ray.init(
        num_cpus=2,
        log_to_driver=False,
        include_dashboard=False,
        ignore_reinit_error=True,
    )
    yield
    ray.shutdown()


# --------------------------------------------------------------------------- #
# Worker function                                                              #
# --------------------------------------------------------------------------- #

def parse_skillcorner_batch(batch):
    """Ray map_batches worker. ``batch`` is a dict-of-arrays / pa.Table-like
    (depending on ``batch_format``). For each row, parse one match and
    concatenate tracking results.
    """
    out_tables = []
    for raw, meta in zip(batch["tracking_bytes"], batch["meta_bytes"]):
        # batch entries arrive as numpy bytes_ / bytes; either works for arrow engine.
        raw_b = bytes(raw)
        meta_b = bytes(meta)
        ds = skillcorner.load_tracking(
            raw_b, meta_b,
            engine="arrow[spark]",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        out_tables.append(ds.tracking)
    combined = pa.concat_tables(out_tables)
    return combined


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

class TestRayMapBatches:

    def test_single_match(self, raw_bytes, meta_bytes, baseline_height):
        matches_ds = ray_data.from_items([
            {"match_id": "m1", "tracking_bytes": raw_bytes, "meta_bytes": meta_bytes}
        ])
        tracking_ds = matches_ds.map_batches(
            parse_skillcorner_batch, batch_format="pyarrow",
        )
        assert tracking_ds.count() == baseline_height

    def test_three_matches_parallel(self, raw_bytes, meta_bytes, baseline_height):
        matches_ds = ray_data.from_items([
            {"match_id": f"m{i}", "tracking_bytes": raw_bytes, "meta_bytes": meta_bytes}
            for i in range(3)
        ])
        tracking_ds = matches_ds.map_batches(
            parse_skillcorner_batch, batch_format="pyarrow",
        )
        assert tracking_ds.count() == 3 * baseline_height
