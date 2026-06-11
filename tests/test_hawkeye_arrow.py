"""Tests for HawkEye arrow + per-slice support.

HawkEye has a multi-file-list + filename-as-metadata shape. The arrow API
uses a `period`+`minute` kwarg toggle to distinguish single-file (per-slice,
for distributed compute) from multi-file (full-match) modes. Input shape
restrictions:
- Arrow engines accept bytes-only inputs (no FileLike) — kloppy-free contract.
- Polars/pyspark engines accept FileLike for driver-side ergonomics.

Coverage here:
- Multi-file (no period/minute): list of (period, minute, bytes) triples on all 4 engines.
- Single-file (period+minute kwargs): bytes on all 4 engines; single FileLike on polars/pyspark only.
- Input contract: arrow rejects FileLike, polars accepts it; mismatched modes raise clear errors.
- Layout matrix: long, long_ball pass; wide → NotImplementedError on arrow.
- Dialect: arrow vs arrow[spark] dtype differences.
- Schema parity: dataset.schemas.tracking matches dataset.tracking.schema.
- **Round-trip parity (critical):** concat per-slice arrow calls == full-match polars baseline.

Gated on pyarrow being installed.
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest

pa = pytest.importorskip("pyarrow")

from fastforward import hawkeye
from fastforward._dataset import TrackingDataset
from tests.config import HE_BALL_FILES, HE_PLAYER_FILES, HE_META_JSON


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _read(p):
    with open(p, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def ball_triples():
    """[(period, minute, bytes), ...] for all ball files."""
    out = []
    for p in HE_BALL_FILES:
        # filename pattern hawkeye_<period>_<minute>.football.samples.ball
        stem = Path(p).name
        parts = stem.replace("hawkeye_", "").split(".")[0].split("_")
        period, minute = int(parts[0]), int(parts[1])
        out.append((period, minute, _read(p)))
    return out


@pytest.fixture(scope="module")
def player_triples():
    out = []
    for p in HE_PLAYER_FILES:
        stem = Path(p).name
        parts = stem.replace("hawkeye_", "").split(".")[0].split("_")
        period, minute = int(parts[0]), int(parts[1])
        out.append((period, minute, _read(p)))
    return out


@pytest.fixture(scope="module")
def meta_bytes():
    return _read(HE_META_JSON)


@pytest.fixture(scope="module")
def polars_dataset_full():
    """Baseline: full match via polars FileLike path."""
    return hawkeye.load_tracking(
        HE_BALL_FILES, HE_PLAYER_FILES, HE_META_JSON, engine="polars",
    )


# --------------------------------------------------------------------------- #
# Multi-file mode shape + parity                                               #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowMultiFile:

    def test_arrow_full_match_returns_arrow_tables(self, ball_triples, player_triples, meta_bytes):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples,
            player_data=player_triples,
            meta_data=meta_bytes,
            engine="arrow",
        )
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)
        assert isinstance(ds.metadata, pa.Table)

    def test_arrow_full_match_row_count_matches_polars(
        self, ball_triples, player_triples, meta_bytes, polars_dataset_full
    ):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples,
            player_data=player_triples,
            meta_data=meta_bytes,
            engine="arrow",
        )
        assert ds.tracking.num_rows == polars_dataset_full.tracking.height

    def test_polars_full_match_via_triples(self, ball_triples, player_triples, meta_bytes, polars_dataset_full):
        """Polars engine should also accept the triples shape (engine-uniform)."""
        ds = hawkeye.load_tracking(
            ball_data=ball_triples,
            player_data=player_triples,
            meta_data=meta_bytes,
            engine="polars",
        )
        assert ds.tracking.height == polars_dataset_full.tracking.height


# --------------------------------------------------------------------------- #
# Single-file (per-slice) mode                                                 #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowSingleFile:
    """Per-slice mode triggered by period+minute kwargs. Inputs are single
    bytes-like (any engine) or single FileLike (polars/pyspark only)."""

    def test_arrow_single_minute(self, ball_triples, player_triples, meta_bytes):
        # Take the first minute
        p, m, ball_b = ball_triples[0]
        _, _, player_b = player_triples[0]
        ds = hawkeye.load_tracking(
            ball_data=ball_b,
            player_data=player_b,
            meta_data=meta_bytes,
            period=p, minute=m,
            engine="arrow",
            include_game_id="match_uuid_42",
        )
        assert isinstance(ds.tracking, pa.Table)
        assert ds.tracking.num_rows > 0
        # All rows should carry the match_id we passed
        if "game_id" in ds.tracking.column_names:
            ids = set(ds.tracking["game_id"].to_pylist())
            assert ids == {"match_uuid_42"}

    def test_polars_single_minute_via_bytes(self, ball_triples, player_triples, meta_bytes):
        p, m, ball_b = ball_triples[0]
        _, _, player_b = player_triples[0]
        ds = hawkeye.load_tracking(
            ball_data=ball_b,
            player_data=player_b,
            meta_data=meta_bytes,
            period=p, minute=m,
            engine="polars",
        )
        assert ds.tracking.height > 0

    def test_polars_single_minute_via_filelike(self):
        ds = hawkeye.load_tracking(
            ball_data=HE_BALL_FILES[0],
            player_data=HE_PLAYER_FILES[0],
            meta_data=HE_META_JSON,
            period=1, minute=1,
            engine="polars",
        )
        assert ds.tracking.height > 0

    def test_pyspark_single_minute_via_bytes(self, ball_triples, player_triples, meta_bytes):
        """Single-file mode must also work on engine='pyspark'."""
        pyspark = pytest.importorskip("pyspark")
        p, m, ball_b = ball_triples[0]
        _, _, player_b = player_triples[0]
        ds = hawkeye.load_tracking(
            ball_data=ball_b,
            player_data=player_b,
            meta_data=meta_bytes,
            period=p, minute=m,
            engine="pyspark",
        )
        # pyspark DataFrame; count() returns int rather than .height
        assert ds.tracking.count() > 0

    def test_include_game_id_false_omits_column_per_slice(self, ball_triples, player_triples, meta_bytes):
        """Per-slice mode must honor include_game_id=False so unioning per-slice
        results across matches produces schema-consistent rows (no per-call drift)."""
        p, m, ball_b = ball_triples[0]
        _, _, player_b = player_triples[0]
        ds = hawkeye.load_tracking(
            ball_data=ball_b,
            player_data=player_b,
            meta_data=meta_bytes,
            period=p, minute=m,
            include_game_id=False,
            engine="arrow",
        )
        assert "game_id" not in ds.tracking.column_names

    def test_include_game_id_str_overrides_per_slice(self, ball_triples, player_triples, meta_bytes):
        """Per-slice mode must propagate include_game_id=str into the tracking table."""
        p, m, ball_b = ball_triples[0]
        _, _, player_b = player_triples[0]
        ds = hawkeye.load_tracking(
            ball_data=ball_b,
            player_data=player_b,
            meta_data=meta_bytes,
            period=p, minute=m,
            include_game_id="custom_match_id",
            engine="arrow",
        )
        ids = set(ds.tracking["game_id"].to_pylist())
        assert ids == {"custom_match_id"}


# --------------------------------------------------------------------------- #
# Mode toggle + error cases                                                    #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowInvalidPeriod:
    """compute_hawkeye_frame_id only supports periods 1-4 (regulation +
    extra time). Any period_id outside that range must fail loudly with
    ValueError rather than silently mis-attributing the within-period
    minute to the period offset."""

    def test_period_5_raises_value_error_single_file(
        self, ball_triples, player_triples, meta_bytes
    ):
        _, _, ball_b = ball_triples[0]
        _, _, player_b = player_triples[0]
        with pytest.raises(ValueError, match="period_id=5"):
            hawkeye.load_tracking(
                ball_data=ball_b, player_data=player_b, meta_data=meta_bytes,
                period=5, minute=1, engine="arrow",
            )


class TestHawkeyeArrowModeToggle:

    def test_period_without_minute_raises(self, ball_triples, player_triples, meta_bytes):
        with pytest.raises(ValueError, match="period and minute must be provided together"):
            hawkeye.load_tracking(
                ball_data=b"placeholder", player_data=b"placeholder",
                meta_data=meta_bytes,
                period=1,  # no minute
                engine="arrow",
            )

    def test_minute_without_period_raises(self, meta_bytes):
        with pytest.raises(ValueError, match="period and minute must be provided together"):
            hawkeye.load_tracking(
                ball_data=b"placeholder", player_data=b"placeholder",
                meta_data=meta_bytes,
                minute=1,  # no period
                engine="arrow",
            )

    def test_single_file_with_list_raises(self, ball_triples, player_triples, meta_bytes):
        with pytest.raises(TypeError, match="single-shaped"):
            hawkeye.load_tracking(
                ball_data=ball_triples,  # list, but single-file mode
                player_data=player_triples,
                meta_data=meta_bytes,
                period=1, minute=1,
                engine="arrow",
            )


# --------------------------------------------------------------------------- #
# Arrow rejects FileLike (kloppy-free contract)                                #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowKloppyFreeContract:
    """Arrow engines must NOT accept FileLike inputs — preserves the
    no-kloppy-on-workers guarantee."""

    def test_arrow_rejects_filelike_meta(self, ball_triples, player_triples):
        with pytest.raises(TypeError, match="kloppy-free"):
            hawkeye.load_tracking(
                ball_data=ball_triples,
                player_data=player_triples,
                meta_data=HE_META_JSON,  # str path — kloppy would be needed
                engine="arrow",
            )

    def test_arrow_rejects_filelike_list_ball_data(self, player_triples, meta_bytes):
        with pytest.raises(TypeError, match="kloppy-free"):
            hawkeye.load_tracking(
                ball_data=HE_BALL_FILES,  # list of paths
                player_data=player_triples,
                meta_data=meta_bytes,
                engine="arrow",
            )

    def test_arrow_rejects_filelike_single_file_mode(self, meta_bytes):
        with pytest.raises(TypeError, match="kloppy-free"):
            hawkeye.load_tracking(
                ball_data=HE_BALL_FILES[0],  # str path in single-file mode
                player_data=HE_PLAYER_FILES[0],
                meta_data=meta_bytes,
                period=1, minute=1,
                engine="arrow",
            )


# --------------------------------------------------------------------------- #
# Round-trip parity (the critical Risk #1 check)                               #
# --------------------------------------------------------------------------- #

class TestHawkeyeRoundTripParity:
    """The gating test: concat of per-slice arrow calls must match the
    full-match polars baseline after sorting."""

    def test_per_slice_concat_matches_full_match(
        self, ball_triples, player_triples, meta_bytes, polars_dataset_full
    ):
        # Call per-slice for each minute, collect tracking tables, concat.
        per_slice_tables = []
        for (p, m, ball_b), (_, _, player_b) in zip(ball_triples, player_triples):
            ds = hawkeye.load_tracking(
                ball_data=ball_b,
                player_data=player_b,
                meta_data=meta_bytes,
                period=p, minute=m,
                engine="arrow",
            )
            per_slice_tables.append(ds.tracking)

        concat = pa.concat_tables(per_slice_tables)
        # Convert via polars for sorting (pyarrow can't sort binary_view columns).
        concat_pl = pl.from_arrow(concat).sort([
            "period_id", "frame_id", "team_id", "player_id",
        ])
        baseline = polars_dataset_full.tracking.sort([
            "period_id", "frame_id", "team_id", "player_id",
        ])

        # Row count parity
        assert concat_pl.height == baseline.height, (
            f"per-slice concat: {concat_pl.height} rows, "
            f"full-match polars: {baseline.height}"
        )
        # Frame_id set parity (rules out shifted/duplicated IDs)
        assert sorted(set(concat_pl["frame_id"].to_list())) == sorted(set(baseline["frame_id"].to_list()))
        # Value-column parity on ALL rows (not a head() spot-check). eq_missing
        # handles NaN==NaN so float columns with sentinel NaNs don't false-fail.
        for col in ("x", "y", "z", "timestamp", "ball_state"):
            if col in concat_pl.columns and col in baseline.columns:
                assert concat_pl[col].eq_missing(baseline[col]).all(), (
                    f"column {col!r} differs"
                )

    def test_per_slice_concat_matches_full_match_period_2(
        self, ball_triples, player_triples, meta_bytes, polars_dataset_full
    ):
        """Period 2 only — guards against bugs that the period-1-dominated
        head() sort order in the prior test would silently pass through."""
        per_slice_tables = []
        for (p, m, ball_b), (_, _, player_b) in zip(ball_triples, player_triples):
            if p != 2:
                continue
            ds = hawkeye.load_tracking(
                ball_data=ball_b, player_data=player_b, meta_data=meta_bytes,
                period=p, minute=m, engine="arrow",
            )
            per_slice_tables.append(ds.tracking)

        if not per_slice_tables:
            pytest.skip("No period 2 files in fixture")

        concat = pa.concat_tables(per_slice_tables)
        concat_pl = pl.from_arrow(concat).sort(
            ["period_id", "frame_id", "team_id", "player_id"]
        )
        baseline = polars_dataset_full.tracking.filter(
            pl.col("period_id") == 2
        ).sort(["period_id", "frame_id", "team_id", "player_id"])

        assert concat_pl.height == baseline.height
        assert sorted(set(concat_pl["frame_id"].to_list())) == sorted(set(baseline["frame_id"].to_list()))
        for col in ("x", "y", "z", "timestamp", "ball_state"):
            if col in concat_pl.columns and col in baseline.columns:
                assert concat_pl[col].eq_missing(baseline[col]).all(), (
                    f"period 2 column {col!r} differs"
                )


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_full_match(self, ball_triples, player_triples, meta_bytes, layout):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples,
            player_data=player_triples,
            meta_data=meta_bytes,
            engine="arrow",
            layout=layout,
        )
        assert ds.tracking.num_rows > 0

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_single_file(self, ball_triples, player_triples, meta_bytes, layout):
        p, m, ball_b = ball_triples[0]
        _, _, player_b = player_triples[0]
        ds = hawkeye.load_tracking(
            ball_data=ball_b, player_data=player_b, meta_data=meta_bytes,
            period=p, minute=m,
            engine="arrow", layout=layout,
        )
        assert ds.tracking.num_rows > 0

    def test_wide_layout_rejected_full_match(self, ball_triples, player_triples, meta_bytes):
        with pytest.raises(NotImplementedError, match="wide"):
            hawkeye.load_tracking(
                ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
                engine="arrow", layout="wide",
            )


# --------------------------------------------------------------------------- #
# Arrow vs arrow[spark] dialect                                                #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowSparkDialect:

    def test_arrow_uses_string_view(self, ball_triples, player_triples, meta_bytes):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow",
        )
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_arrow_spark_uses_string(self, ball_triples, player_triples, meta_bytes):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow[spark]",
        )
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)

    def test_arrow_timestamp_is_duration_ms(self, ball_triples, player_triples, meta_bytes):
        """engine='arrow' uses Polars-style duration[ms] for timestamps."""
        ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow",
        )
        ts = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_duration(ts) and ts.unit == "ms"

    def test_arrow_spark_timestamp_is_int64(self, ball_triples, player_triples, meta_bytes):
        """engine='arrow[spark]' normalizes duration[ms] to int64 for Spark."""
        ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow[spark]",
        )
        ts = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_int64(ts)


# --------------------------------------------------------------------------- #
# Schema factory                                                               #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowSchemas:

    def test_schemas_factory_matches_dataset_schema(self, ball_triples, player_triples, meta_bytes):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow[spark]",
        )
        s = hawkeye.schemas(layout="long", engine="arrow[spark]")
        assert s.tracking == ds.tracking.schema

    def test_dataset_schemas_property_matches_factory(self, ball_triples, player_triples, meta_bytes):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow[spark]",
        )
        factory = hawkeye.schemas(layout="long", engine="arrow[spark]")
        assert ds.schemas.tracking == factory.tracking
        assert ds.schemas.tracking_spark == factory.tracking_spark

    def test_wide_layout_schemas_raises(self):
        s = hawkeye.schemas(layout="wide", engine="arrow[spark]")
        with pytest.raises(NotImplementedError):
            _ = s.tracking

    def test_pyspark_struct_type_available(self):
        pyspark = pytest.importorskip("pyspark")
        s = hawkeye.schemas(layout="long", engine="arrow[spark]")
        from pyspark.sql.types import StructType
        assert isinstance(s.tracking_spark, StructType)
        first = s.tracking_spark.fields[0]
        assert first.name == "game_id"


# --------------------------------------------------------------------------- #
# Post-load transforms on arrow                                                #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowTransforms:
    """Confirms `dataset.transform(...)` runs against pyarrow.Table inputs and
    produces values equal to the polars-engine baseline transform."""

    def test_transform_coords_and_orientation_in_rust(
        self, ball_triples, player_triples, meta_bytes
    ):
        arrow_ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow",
        )
        polars_ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="polars",
        )
        arrow_t = arrow_ds.transform(
            to_coordinates="opta", to_orientation="static_away_home",
        )
        polars_t = polars_ds.transform(
            to_coordinates="opta", to_orientation="static_away_home",
        )

        assert arrow_t.engine == "arrow"
        assert isinstance(arrow_t.tracking, pa.Table)

        # Sort both via polars so row ordering matches; transform must agree on
        # x/y/z (orientation flips coords, coordinates rescales them).
        arrow_pl = pl.from_arrow(arrow_t.tracking).sort(
            ["period_id", "frame_id", "team_id", "player_id"]
        )
        polars_sorted = polars_t.tracking.sort(
            ["period_id", "frame_id", "team_id", "player_id"]
        )
        for col in ("x", "y", "z"):
            a = arrow_pl[col]
            p = polars_sorted[col]
            # diff is NaN where either input is NaN (e.g., missing position);
            # treat NaN-vs-NaN as "agree." Real divergence shows up as finite > tol.
            diffs = (a - p).abs()
            ok = diffs.is_null() | diffs.is_nan() | (diffs < 1e-3)
            assert ok.all(), f"transform diverged on column {col}"


# --------------------------------------------------------------------------- #
# Engine converters round-trip on hawkeye data                                 #
# --------------------------------------------------------------------------- #

class TestHawkeyeEngineConverters:
    """Per-provider sanity checks that `to_arrow / to_polars / to_pyspark`
    work on a hawkeye dataset (the cross-cutting infra is exercised in
    test_arrow_output.py on skillcorner, but provider-specific fixtures may
    have schema quirks the cross-cutting tests don't see)."""

    def test_arrow_to_polars(self, ball_triples, player_triples, meta_bytes, polars_dataset_full):
        arrow_ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow",
        )
        converted = arrow_ds.to_polars()
        assert converted.engine == "polars"
        assert isinstance(converted.tracking, pl.DataFrame)
        assert converted.tracking.height == polars_dataset_full.tracking.height

    def test_arrow_polars_arrow_roundtrip(self, ball_triples, player_triples, meta_bytes):
        original = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow",
        )
        via_polars = original.to_polars().to_arrow()
        assert via_polars.engine == "arrow"
        assert isinstance(via_polars.tracking, pa.Table)
        assert via_polars.tracking.num_rows == original.tracking.num_rows
        assert via_polars.tracking.column_names == original.tracking.column_names

    def test_polars_to_arrow_to_polars(self, ball_triples, player_triples, meta_bytes):
        original = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="polars",
        )
        via_arrow = original.to_arrow().to_polars()
        assert via_arrow.engine == "polars"
        assert isinstance(via_arrow.tracking, pl.DataFrame)
        assert via_arrow.tracking.height == original.tracking.height
        assert via_arrow.tracking.columns == original.tracking.columns

    def test_to_arrow_idempotent(self, ball_triples, player_triples, meta_bytes):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow",
        )
        assert ds.to_arrow() is ds

    def test_arrow_to_pyspark(self, ball_triples, player_triples, meta_bytes, polars_dataset_full):
        pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession
        from pyspark.sql import DataFrame as SparkDataFrame

        spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("ff-hawkeye-arrow-to-spark")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        try:
            arrow_ds = hawkeye.load_tracking(
                ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
                engine="arrow",
            )
            converted = arrow_ds.to_pyspark(spark)
            assert converted.engine == "pyspark"
            assert isinstance(converted.tracking, SparkDataFrame)
            assert converted.tracking.count() == polars_dataset_full.tracking.height
        finally:
            spark.stop()


# --------------------------------------------------------------------------- #
# include_officials on the arrow path                                          #
# --------------------------------------------------------------------------- #

class TestHawkeyeArrowIncludeOfficials:
    """`include_officials` adds rows (team_id='officials') but no new columns,
    so it's a row-set toggle that needs arrow-path coverage. The polars
    coverage in tests/test_hawkeye.py confirms the Rust parsing is correct;
    these tests confirm the row count flows through the arrow path."""

    def test_arrow_default_excludes_officials(self, ball_triples, player_triples, meta_bytes):
        ds = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow",
        )
        team_ids = set(ds.tracking.column("team_id").to_pylist())
        assert "officials" not in team_ids

    def test_arrow_include_officials_adds_rows(self, ball_triples, player_triples, meta_bytes):
        no_off = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow", include_officials=False,
        )
        with_off = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow", include_officials=True,
        )
        assert with_off.tracking.num_rows > no_off.tracking.num_rows
        team_ids = set(with_off.tracking.column("team_id").to_pylist())
        assert "officials" in team_ids

    def test_arrow_schema_unchanged_by_include_officials(
        self, ball_triples, player_triples, meta_bytes
    ):
        """Officials add rows, not columns — column set must be identical."""
        no_off = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow", include_officials=False,
        )
        with_off = hawkeye.load_tracking(
            ball_data=ball_triples, player_data=player_triples, meta_data=meta_bytes,
            engine="arrow", include_officials=True,
        )
        assert no_off.tracking.schema == with_off.tracking.schema
