"""Shared helpers for per-provider arrow-engine test classes.

These exist to cut ~75 lines of identical body per provider into ~3-line
test methods. The canonical pattern they were extracted from lives in
[tests/test_scisports_arrow.py](tests/test_scisports_arrow.py).

Usage at a test site:

    from tests._arrow_helpers import (
        assert_arrow_transform_matches_polars,
        assert_arrow_to_polars_height_match,
        assert_arrow_polars_arrow_roundtrip,
        assert_polars_to_arrow_to_polars,
        assert_to_arrow_idempotent,
        assert_arrow_accepts_bytes_like,
        assert_arrow_rejects_paths,
    )

    class TestCdfArrowTransforms:
        def test_transform(self, raw_bytes, meta_bytes):
            arrow_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
            polars_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="polars")
            assert_arrow_transform_matches_polars(arrow_ds, polars_ds)

    class TestCdfArrowInputContract:
        def test_accepts_bytes_like(self, raw_bytes, meta_bytes):
            assert_arrow_accepts_bytes_like(
                lambda r, m: cdf.load_tracking(r, m, engine="arrow"),
                raw_bytes, meta_bytes,
            )

        def test_path_string_rejected(self):
            assert_arrow_rejects_paths(
                lambda r, m: cdf.load_tracking(r, m, engine="arrow"),
                CDF_RAW, CDF_META,
            )

Providers with non-standard signatures (gradientsports adds roster_bytes,
respovision uses a filename= kwarg) wire via the lambda at the call site.
"""

from __future__ import annotations

import io
from typing import Callable, Iterable

import polars as pl
import pyarrow as pa
import pytest

from fastforward._dataset import TrackingDataset


# --------------------------------------------------------------------------- #
# Post-load transforms                                                         #
# --------------------------------------------------------------------------- #


def assert_arrow_transform_matches_polars(
    arrow_ds: TrackingDataset,
    polars_ds: TrackingDataset,
    *,
    to_coordinates: str = "opta",
    to_orientation: str = "static_away_home",
    cols: Iterable[str] = ("x", "y", "z"),
    sort_keys: Iterable[str] = ("period_id", "frame_id", "team_id", "player_id"),
    tol: float = 1e-3,
) -> None:
    """Apply the same transform to both inputs, assert x/y/z agree.

    NaN-aware: a diff that is null or NaN (one of the inputs was missing)
    counts as agreement; only finite-and-large differences fail.

    Lesson from the hawkeye/signality QA pass: f32 paths sometimes drift by
    < 1e-4 across the polars vs arrow path; tol is generous.
    """
    arrow_t = arrow_ds.transform(to_coordinates=to_coordinates, to_orientation=to_orientation)
    polars_t = polars_ds.transform(to_coordinates=to_coordinates, to_orientation=to_orientation)

    assert arrow_t.engine == "arrow", (
        f"transform on arrow dataset switched engine to {arrow_t.engine!r}"
    )
    assert isinstance(arrow_t.tracking, pa.Table)

    # Guard against trivially-passing on degenerate inputs (empty fixture
    # or a layout that strips every comparable column).
    assert arrow_t.tracking.num_rows > 0, "arrow transform produced zero rows"
    assert polars_t.tracking.height > 0, "polars transform produced zero rows"
    cols_present = [c for c in cols if c in arrow_t.tracking.column_names
                    and c in polars_t.tracking.columns]
    assert cols_present, (
        f"none of cols={list(cols)} present in both schemas — test would "
        f"trivially pass"
    )

    sort_keys_list = [k for k in sort_keys if k in arrow_t.tracking.column_names]
    arrow_pl = pl.from_arrow(arrow_t.tracking).sort(sort_keys_list)
    polars_sorted = polars_t.tracking.sort(sort_keys_list)

    for col in cols_present:
        a = arrow_pl[col]
        p = polars_sorted[col]
        diffs = (a - p).abs()
        ok = diffs.is_null() | diffs.is_nan() | (diffs < tol)
        assert ok.all(), f"transform diverged on column {col}"


# --------------------------------------------------------------------------- #
# Engine-converter round-trips                                                 #
# --------------------------------------------------------------------------- #


def assert_arrow_to_polars_height_match(
    arrow_ds: TrackingDataset, polars_ds: TrackingDataset
) -> None:
    """arrow_ds.to_polars() converts cleanly and matches the polars baseline."""
    converted = arrow_ds.to_polars()
    assert converted.engine == "polars"
    assert isinstance(converted.tracking, pl.DataFrame)
    assert converted.tracking.height == polars_ds.tracking.height


def assert_arrow_polars_arrow_roundtrip(arrow_ds: TrackingDataset) -> None:
    """arrow → polars → arrow preserves row count, column names, AND values.

    A weaker version (rows + names only) was caught silently passing all-null
    converters in the Phase B QA review. Always compare values now.
    """
    via_polars = arrow_ds.to_polars().to_arrow()
    assert via_polars.engine == "arrow"
    assert isinstance(via_polars.tracking, pa.Table)
    assert via_polars.tracking.num_rows == arrow_ds.tracking.num_rows
    assert via_polars.tracking.column_names == arrow_ds.tracking.column_names
    # Value-equality column by column. .equals() on full tables is too strict
    # because pyarrow records chunk layout in equality; iterate columns to
    # compare values regardless of chunking.
    for col in arrow_ds.tracking.column_names:
        assert via_polars.tracking[col].to_pylist() == arrow_ds.tracking[col].to_pylist(), (
            f"arrow → polars → arrow diverged on column {col!r}"
        )


def assert_polars_to_arrow_to_polars(polars_ds: TrackingDataset) -> None:
    """polars → arrow → polars preserves row count, column names, AND values.

    NaN-aware: NaN-vs-NaN counts as equal (Polars' `equals(null_equal=True)`).
    """
    via_arrow = polars_ds.to_arrow().to_polars()
    assert via_arrow.engine == "polars"
    assert isinstance(via_arrow.tracking, pl.DataFrame)
    assert via_arrow.tracking.height == polars_ds.tracking.height
    assert via_arrow.tracking.columns == polars_ds.tracking.columns
    # Polars equals() with null_equal=True treats null-vs-null and NaN-vs-NaN
    # as equal — the right semantics for a pure roundtrip.
    assert via_arrow.tracking.equals(polars_ds.tracking, null_equal=True), (
        "polars → arrow → polars diverged on values"
    )


def assert_to_arrow_idempotent(arrow_ds: TrackingDataset) -> None:
    """to_arrow() on an already-arrow dataset returns self."""
    assert arrow_ds.to_arrow() is arrow_ds


# --------------------------------------------------------------------------- #
# Input contract — bytes-only on arrow engines (kloppy-free)                   #
# --------------------------------------------------------------------------- #


def assert_arrow_accepts_bytes_like(
    load_fn: Callable[..., TrackingDataset],
    *byte_args: bytes,
) -> None:
    """Call `load_fn(*byte_args)` four times, wrapping every positional arg
    in turn as bytes / bytearray / memoryview / BytesIO. Each call should
    yield a non-empty dataset.

    For providers whose signature varies (extras like roster_data, kwargs
    like filename=), wire via a lambda at the call site:

        assert_arrow_accepts_bytes_like(
            lambda r, m, rb: gradientsports.load_tracking(r, m, rb, engine="arrow"),
            raw_bytes, meta_bytes, roster_bytes,
        )
    """
    wrappers: list[tuple[str, Callable[[bytes], object]]] = [
        ("bytes", bytes),
        ("bytearray", bytearray),
        ("memoryview", memoryview),
        ("BytesIO", io.BytesIO),
    ]
    for name, wrapper in wrappers:
        wrapped = [wrapper(b) for b in byte_args]
        ds = load_fn(*wrapped)
        assert ds.tracking.num_rows > 0, f"empty result with input form {name!r}"
        # Catch silent fall-back to a non-arrow engine — without this, a
        # bug that opened a FileLike and went through the polars path could
        # pass.
        assert ds.engine in ("arrow", "arrow[spark]"), (
            f"input form {name!r} loaded but engine is {ds.engine!r}, "
            f"expected arrow / arrow[spark]"
        )
        assert isinstance(ds.tracking, pa.Table), (
            f"input form {name!r} loaded but tracking is {type(ds.tracking).__name__}, "
            f"expected pa.Table"
        )


def assert_arrow_rejects_paths(
    load_fn: Callable[..., TrackingDataset],
    *path_args: str,
) -> None:
    """Calling load_fn with path-string args should raise TypeError on the
    arrow engine (kloppy-free contract). Pins the contract via match= so a
    stray TypeError (wrong positional count, etc) doesn't slip through.
    """
    with pytest.raises(TypeError, match=r"(?i)bytes|kloppy-free|file-like"):
        load_fn(*path_args)


def assert_arrow_accepts_buffered_reader(
    load_fn: Callable[..., TrackingDataset],
    tmp_path,
    *byte_args: bytes,
) -> None:
    """Write each bytes arg to a file under `tmp_path` and pass
    `open(path, "rb")` (BufferedReader, the result of the most common way
    a user opens a file) to `load_fn`. The kloppy-free `_to_bytes` contract
    documents BufferedReader as accepted; this test pins that promise.

    `tmp_path` is the standard pytest fixture; pass it in at the call site.
    """
    paths = [tmp_path / f"input_{i}.bin" for i in range(len(byte_args))]
    for path, b in zip(paths, byte_args):
        path.write_bytes(b)
    handles = [open(p, "rb") for p in paths]
    try:
        ds = load_fn(*handles)
    finally:
        for h in handles:
            h.close()
    assert ds.tracking.num_rows > 0, "empty result with BufferedReader input"
    assert ds.engine in ("arrow", "arrow[spark]"), (
        f"BufferedReader input loaded but engine is {ds.engine!r}"
    )
    assert isinstance(ds.tracking, pa.Table), (
        f"BufferedReader input loaded but tracking is {type(ds.tracking).__name__}"
    )


def assert_arrow_accepts_gzip_stream(
    load_fn: Callable[..., TrackingDataset],
    *byte_args: bytes,
) -> None:
    """Wrap each bytes arg in an in-memory gzipped `GzipFile` (read mode)
    and pass to `load_fn`. Reading a GzipFile returns the original
    decompressed bytes, so a correct parser sees identical input.

    The kloppy-free `_to_bytes` contract documents gzip.GzipFile as accepted
    (any binary-mode `io.IOBase`); this test pins that promise.
    """
    import gzip
    handles = [
        gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(b)))
        for b in byte_args
    ]
    try:
        ds = load_fn(*handles)
    finally:
        for h in handles:
            h.close()
    assert ds.tracking.num_rows > 0, "empty result with GzipFile input"
    assert ds.engine in ("arrow", "arrow[spark]"), (
        f"GzipFile input loaded but engine is {ds.engine!r}"
    )
    assert isinstance(ds.tracking, pa.Table), (
        f"GzipFile input loaded but tracking is {type(ds.tracking).__name__}"
    )


# --------------------------------------------------------------------------- #
# Atomic dialect / schemas helpers (D.2 refactor)                              #
# --------------------------------------------------------------------------- #
# These are one-purpose helpers. Each replaces the body of exactly ONE test
# method in a provider's TestProviderArrowSparkDialect or TestProviderArrowSchemas
# class. Test granularity stays intact: each existing test method becomes
# 2 lines (load + one helper call), and a failure points at the exact
# assertion that broke.


def assert_arrow_engine_uses_string_view(ds: TrackingDataset) -> None:
    """engine='arrow' uses Polars-style string_view for string columns."""
    team_id_t = ds.tracking.schema.field("team_id").type
    assert pa.types.is_string_view(team_id_t), (
        f"expected string_view under engine='arrow', got {team_id_t}"
    )


def assert_arrow_spark_engine_uses_string(ds: TrackingDataset) -> None:
    """engine='arrow[spark]' normalizes string_view → plain string."""
    team_id_t = ds.tracking.schema.field("team_id").type
    assert (
        pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)
    ), f"expected plain string under engine='arrow[spark]', got {team_id_t}"


def assert_arrow_engine_timestamp_duration_ms(ds: TrackingDataset) -> None:
    """engine='arrow' keeps Polars-style duration[ms] for the timestamp column."""
    ts_t = ds.tracking.schema.field("timestamp").type
    assert pa.types.is_duration(ts_t) and ts_t.unit == "ms", (
        f"expected duration[ms] under engine='arrow', got {ts_t}"
    )


def assert_arrow_spark_engine_timestamp_int64(ds: TrackingDataset) -> None:
    """engine='arrow[spark]' normalizes duration[ms] → int64 for Spark."""
    ts_t = ds.tracking.schema.field("timestamp").type
    assert pa.types.is_int64(ts_t), (
        f"expected int64 under engine='arrow[spark]', got {ts_t}"
    )


def assert_schemas_factory_matches_dataset(factory, ds: TrackingDataset) -> None:
    """`provider.schemas(...).tracking == ds.tracking.schema` and same for
    metadata, teams, players, periods."""
    assert factory.tracking == ds.tracking.schema, "tracking schema differs"
    assert factory.metadata == ds.metadata.schema, "metadata schema differs"
    assert factory.teams == ds.teams.schema, "teams schema differs"
    assert factory.players == ds.players.schema, "players schema differs"
    assert factory.periods == ds.periods.schema, "periods schema differs"


def assert_dataset_schemas_property_matches_factory(
    ds: TrackingDataset, factory
) -> None:
    """`ds.schemas.tracking == factory.tracking` and same for the spark
    StructType variant."""
    assert ds.schemas.tracking == factory.tracking, (
        "ds.schemas.tracking differs from factory.tracking"
    )
    pytest.importorskip("pyspark")  # tracking_spark builds a real pyspark StructType
    assert ds.schemas.tracking_spark == factory.tracking_spark, (
        "ds.schemas.tracking_spark differs from factory.tracking_spark"
    )


def assert_wide_layout_schemas_raises(wide_factory) -> None:
    """Accessing the tracking schemas on a wide-layout factory raises
    NotImplementedError on both tracking and tracking_spark properties."""
    with pytest.raises(NotImplementedError):
        _ = wide_factory.tracking
    with pytest.raises(NotImplementedError):
        _ = wide_factory.tracking_spark


def assert_pyspark_struct_first_field_is_game_id(factory) -> None:
    """`factory.tracking_spark` is a pyspark StructType with `game_id` as
    the first field (when include_game_id=True, the default).
    """
    pytest.importorskip("pyspark")
    from pyspark.sql.types import StructType
    assert isinstance(factory.tracking_spark, StructType), (
        f"expected StructType, got {type(factory.tracking_spark).__name__}"
    )
    first = factory.tracking_spark.fields[0]
    assert first.name == "game_id", (
        f"expected first field to be 'game_id', got {first.name!r}"
    )
