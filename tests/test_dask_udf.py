"""End-to-end Dask `map_partitions` integration test for engine='arrow'.

Mirrors tests/test_pyspark_udf.py for the Dask framework. Uses a local
single-process Dask DataFrame; no cluster setup required beyond installing
``dask[dataframe]``.

Dask consumes Polars-style Arrow types natively via ``.to_pandas()``, so
``engine="arrow"`` (faithful types) is the right engine here. No
``normalize_arrow_for_spark`` cast in the worker function.

Gated on dask + pandas + pyarrow being installed.
"""

from __future__ import annotations

import pytest

dd = pytest.importorskip("dask.dataframe")
pa = pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

import dask

# Dask 2024+ converts object columns to pyarrow string by default. That casts
# our `bytes` cells to `str`, which the arrow engines reject. Disable for the
# whole test module so binary columns survive `from_pandas`.
dask.config.set({"dataframe.convert-string": False})

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


# --------------------------------------------------------------------------- #
# Worker function                                                              #
# --------------------------------------------------------------------------- #

def parse_skillcorner_partition(df):
    """Dask map_partitions worker. Each partition is a pandas DataFrame with
    (match_id, tracking_bytes, meta_bytes) columns. For each row, parse one
    match and return the concatenated tracking rows as a pandas DataFrame.
    """
    out = []
    for _, row in df.iterrows():
        ds = skillcorner.load_tracking(
            row["tracking_bytes"], row["meta_bytes"],
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        out.append(ds.tracking.to_pandas())
    if not out:
        # Return an empty frame with the right columns so map_partitions meta
        # inference doesn't blow up on an empty partition.
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

class TestDaskMapPartitions:

    def test_single_match(self, raw_bytes, meta_bytes, baseline_height):
        matches_pdf = pd.DataFrame({
            "match_id": ["m1"],
            "tracking_bytes": [raw_bytes],
            "meta_bytes": [meta_bytes],
        })
        matches_ddf = dd.from_pandas(matches_pdf, npartitions=1)
        meta = parse_skillcorner_partition(matches_pdf.head(1))
        result = matches_ddf.map_partitions(
            parse_skillcorner_partition, meta=meta,
        ).compute()
        assert len(result) == baseline_height

    def test_three_matches_parallel(self, raw_bytes, meta_bytes, baseline_height):
        matches_pdf = pd.DataFrame({
            "match_id": [f"m{i}" for i in range(3)],
            "tracking_bytes": [raw_bytes] * 3,
            "meta_bytes": [meta_bytes] * 3,
        })
        # 2 partitions → at least one partition holds 2 matches
        matches_ddf = dd.from_pandas(matches_pdf, npartitions=2)
        meta = parse_skillcorner_partition(matches_pdf.head(1))
        result = matches_ddf.map_partitions(
            parse_skillcorner_partition, meta=meta,
        ).compute()
        assert len(result) == 3 * baseline_height

    def test_values_match_baseline(self, raw_bytes, meta_bytes):
        """Sample a few (frame_id, player_id) rows from the Dask output and
        compare positions to the in-process polars baseline.
        """
        matches_pdf = pd.DataFrame({
            "match_id": ["m1"],
            "tracking_bytes": [raw_bytes],
            "meta_bytes": [meta_bytes],
        })
        matches_ddf = dd.from_pandas(matches_pdf, npartitions=1)
        meta = parse_skillcorner_partition(matches_pdf.head(1))
        result = matches_ddf.map_partitions(
            parse_skillcorner_partition, meta=meta,
        ).compute()

        baseline = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="polars",
            include_ball_owning_player=True,
            include_is_detected=True,
        ).tracking
        baseline_lookup = {
            (row["frame_id"], row["player_id"]): (row["x"], row["y"], row["z"])
            for row in baseline.head(50).iter_rows(named=True)
        }
        sample = result.head(50)
        for _, r in sample.iterrows():
            key = (int(r["frame_id"]), r["player_id"])
            if key not in baseline_lookup:
                continue
            x, y, z = baseline_lookup[key]
            if x is not None and r["x"] is not None:
                assert abs(r["x"] - x) < 1e-3
                assert abs(r["y"] - y) < 1e-3
                assert abs(r["z"] - z) < 1e-3
