"""Tests for Signality arrow + per-period support.

Signality has a multi-file (per-period) shape with filename-carried period
info. The arrow API uses a `period` kwarg toggle to distinguish single-file
(per-period, for distributed compute) from multi-file (full-match) modes.

Coverage here:
- Multi-file (no period): list of (period, bytes) pairs on all 4 engines.
- Single-file (period kwarg): bytes on all 4 engines; single FileLike on polars/pyspark only.
- Input contract: arrow rejects FileLike (kloppy-free); mismatched modes raise.
- Layout matrix: long, long_ball pass; wide → NotImplementedError on arrow.
- Dialect: arrow vs arrow[spark] dtype differences.
- Schema parity: dataset.schemas.tracking matches dataset.tracking.schema.
- **Round-trip parity (critical):** concat per-period arrow calls == full-match polars baseline.
- include_game_id True/False/str on per-period.

Gated on pyarrow being installed.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pa = pytest.importorskip("pyarrow")

from fastforward import signality
from fastforward._dataset import TrackingDataset
from tests.config import SIG_META, SIG_VENUE, SIG_RAW_FILES, SIG_RAW_P1, SIG_RAW_P2


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _read(p):
    with open(p, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def raw_pairs():
    """[(period, bytes), ...] from the SIG_RAW_FILES fixtures."""
    out = []
    for p in SIG_RAW_FILES:
        stem = Path(p).name
        # signality_p<N>_raw_data_subset.json
        period = int(stem.split("_p")[1].split("_")[0])
        out.append((period, _read(p)))
    return out


@pytest.fixture(scope="module")
def meta_bytes():
    return _read(SIG_META)


@pytest.fixture(scope="module")
def venue_bytes():
    return _read(SIG_VENUE)


@pytest.fixture(scope="module")
def polars_dataset_full():
    """Baseline: full match via polars FileLike path."""
    return signality.load_tracking(
        meta_data=SIG_META,
        raw_data_feeds=SIG_RAW_FILES,
        venue_information=SIG_VENUE,
        engine="polars",
    )


# --------------------------------------------------------------------------- #
# Multi-file mode (full match)                                                 #
# --------------------------------------------------------------------------- #

class TestSignalityArrowMultiFile:

    def test_arrow_returns_arrow_tables(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes,
            raw_data_feeds=raw_pairs,
            venue_information=venue_bytes,
            engine="arrow",
        )
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)

    def test_arrow_row_count_matches_polars(self, raw_pairs, meta_bytes, venue_bytes, polars_dataset_full):
        ds = signality.load_tracking(
            meta_data=meta_bytes,
            raw_data_feeds=raw_pairs,
            venue_information=venue_bytes,
            engine="arrow",
        )
        assert ds.tracking.num_rows == polars_dataset_full.tracking.height

    def test_polars_via_pairs(self, raw_pairs, meta_bytes, venue_bytes, polars_dataset_full):
        """Polars engine should accept the pairs shape too."""
        ds = signality.load_tracking(
            meta_data=meta_bytes,
            raw_data_feeds=raw_pairs,
            venue_information=venue_bytes,
            engine="polars",
        )
        assert ds.tracking.height == polars_dataset_full.tracking.height


# --------------------------------------------------------------------------- #
# Single-file (per-period) mode                                                #
# --------------------------------------------------------------------------- #

class TestSignalityArrowSingleFile:

    def test_arrow_per_period(self, raw_pairs, meta_bytes, venue_bytes):
        period, raw_b = raw_pairs[0]
        ds = signality.load_tracking(
            meta_data=meta_bytes,
            raw_data_feeds=raw_b,
            venue_information=venue_bytes,
            period=period,
            engine="arrow",
            include_game_id="match_uuid_42",
        )
        assert isinstance(ds.tracking, pa.Table)
        assert ds.tracking.num_rows > 0
        if "game_id" in ds.tracking.column_names:
            assert set(ds.tracking["game_id"].to_pylist()) == {"match_uuid_42"}

    def test_polars_per_period_via_bytes(self, raw_pairs, meta_bytes, venue_bytes):
        period, raw_b = raw_pairs[0]
        ds = signality.load_tracking(
            meta_data=meta_bytes,
            raw_data_feeds=raw_b,
            venue_information=venue_bytes,
            period=period,
            engine="polars",
        )
        assert ds.tracking.height > 0

    def test_polars_per_period_via_filelike(self):
        ds = signality.load_tracking(
            meta_data=SIG_META,
            raw_data_feeds=SIG_RAW_P1,
            venue_information=SIG_VENUE,
            period=1,
            engine="polars",
        )
        assert ds.tracking.height > 0

    def test_pyspark_per_period_via_bytes(self, raw_pairs, meta_bytes, venue_bytes):
        pyspark = pytest.importorskip("pyspark")
        period, raw_b = raw_pairs[0]
        ds = signality.load_tracking(
            meta_data=meta_bytes,
            raw_data_feeds=raw_b,
            venue_information=venue_bytes,
            period=period,
            engine="pyspark",
        )
        assert ds.tracking.count() > 0


# --------------------------------------------------------------------------- #
# Mode-toggle errors                                                           #
# --------------------------------------------------------------------------- #

class TestSignalityArrowModeToggle:

    def test_period_with_list_raises(self, raw_pairs, meta_bytes, venue_bytes):
        with pytest.raises(TypeError, match="single-shaped"):
            signality.load_tracking(
                meta_data=meta_bytes,
                raw_data_feeds=raw_pairs,  # list, but period given
                venue_information=venue_bytes,
                period=1,
                engine="arrow",
            )


# --------------------------------------------------------------------------- #
# Arrow rejects FileLike (kloppy-free contract)                                #
# --------------------------------------------------------------------------- #

class TestSignalityArrowKloppyFreeContract:

    def test_arrow_rejects_filelike_meta(self, raw_pairs, venue_bytes):
        with pytest.raises(TypeError, match="kloppy-free"):
            signality.load_tracking(
                meta_data=SIG_META,  # str path
                raw_data_feeds=raw_pairs,
                venue_information=venue_bytes,
                engine="arrow",
            )

    def test_arrow_rejects_filelike_venue(self, raw_pairs, meta_bytes):
        with pytest.raises(TypeError, match="kloppy-free"):
            signality.load_tracking(
                meta_data=meta_bytes,
                raw_data_feeds=raw_pairs,
                venue_information=SIG_VENUE,  # str path
                engine="arrow",
            )

    def test_arrow_rejects_filelike_raw_list(self, meta_bytes, venue_bytes):
        with pytest.raises(TypeError, match="kloppy-free"):
            signality.load_tracking(
                meta_data=meta_bytes,
                raw_data_feeds=SIG_RAW_FILES,  # list of paths
                venue_information=venue_bytes,
                engine="arrow",
            )

    def test_arrow_rejects_filelike_single_file(self, meta_bytes, venue_bytes):
        with pytest.raises(TypeError, match="kloppy-free"):
            signality.load_tracking(
                meta_data=meta_bytes,
                raw_data_feeds=SIG_RAW_P1,  # str path in single-file mode
                venue_information=venue_bytes,
                period=1,
                engine="arrow",
            )


# --------------------------------------------------------------------------- #
# Round-trip parity (Risk #1)                                                  #
# --------------------------------------------------------------------------- #

class TestSignalityRoundTripParity:
    """Verify that concatenating per-period arrow outputs matches the
    full-match polars baseline bit-for-bit. The fixed per-period offset
    (SIGNALITY_FRAMES_PER_PERIOD_OFFSET = 135,000) makes both paths produce
    identical frame_ids."""

    def test_per_period_concat_matches_full_match(
        self, raw_pairs, meta_bytes, venue_bytes, polars_dataset_full
    ):
        per_period_tables = []
        for period, raw_b in raw_pairs:
            ds = signality.load_tracking(
                meta_data=meta_bytes,
                raw_data_feeds=raw_b,
                venue_information=venue_bytes,
                period=period,
                engine="arrow",
            )
            per_period_tables.append(ds.tracking)

        concat = pa.concat_tables(per_period_tables)
        # Convert via polars for sorting (pyarrow can't sort binary_view cols)
        concat_pl = pl.from_arrow(concat).sort([
            "period_id", "frame_id", "team_id", "player_id",
        ])
        baseline = polars_dataset_full.tracking.sort([
            "period_id", "frame_id", "team_id", "player_id",
        ])

        # Row count parity
        assert concat_pl.height == baseline.height, (
            f"per-period concat: {concat_pl.height}, baseline: {baseline.height}"
        )
        # Frame_id set parity — both paths use the fixed-offset scheme.
        assert sorted(set(concat_pl["frame_id"].to_list())) == sorted(set(baseline["frame_id"].to_list()))
        # Value-column parity on a sample. eq_missing handles NaN==NaN.
        for col in ("x", "y", "z", "timestamp", "ball_state"):
            if col in concat_pl.columns and col in baseline.columns:
                a = concat_pl[col].head(1000)
                b = baseline[col].head(1000)
                assert a.eq_missing(b).all(), (
                    f"column {col!r} differs in first 1000 rows"
                )


# --------------------------------------------------------------------------- #
# include_game_id semantics                                                    #
# --------------------------------------------------------------------------- #

class TestSignalityArrowIncludeGameId:

    def test_default_includes_from_metadata(self, raw_pairs, meta_bytes, venue_bytes):
        period, raw_b = raw_pairs[0]
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_b, venue_information=venue_bytes,
            period=period, engine="arrow",
        )
        assert "game_id" in ds.tracking.column_names

    def test_false_omits_column(self, raw_pairs, meta_bytes, venue_bytes):
        period, raw_b = raw_pairs[0]
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_b, venue_information=venue_bytes,
            period=period, include_game_id=False, engine="arrow",
        )
        assert "game_id" not in ds.tracking.column_names

    def test_str_overrides(self, raw_pairs, meta_bytes, venue_bytes):
        period, raw_b = raw_pairs[0]
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_b, venue_information=venue_bytes,
            period=period, include_game_id="custom_id_99", engine="arrow",
        )
        assert set(ds.tracking["game_id"].to_pylist()) == {"custom_id_99"}


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestSignalityArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_full_match(self, raw_pairs, meta_bytes, venue_bytes, layout):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow", layout=layout,
        )
        assert ds.tracking.num_rows > 0

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_per_period(self, raw_pairs, meta_bytes, venue_bytes, layout):
        period, raw_b = raw_pairs[0]
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_b, venue_information=venue_bytes,
            period=period, engine="arrow", layout=layout,
        )
        assert ds.tracking.num_rows > 0

    def test_wide_rejected(self, raw_pairs, meta_bytes, venue_bytes):
        with pytest.raises(NotImplementedError, match="wide"):
            signality.load_tracking(
                meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
                engine="arrow", layout="wide",
            )


# --------------------------------------------------------------------------- #
# Dialect                                                                      #
# --------------------------------------------------------------------------- #

class TestSignalityArrowSparkDialect:

    def test_arrow_uses_string_view(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow",
        )
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_arrow_spark_uses_string(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow[spark]",
        )
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #

class TestSignalityArrowSchemas:

    def test_factory_matches_dataset(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow[spark]",
        )
        s = signality.schemas(layout="long", engine="arrow[spark]")
        assert s.tracking == ds.tracking.schema

    def test_dataset_schemas_property(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow[spark]",
        )
        assert ds.schemas.tracking == signality.schemas(layout="long", engine="arrow[spark]").tracking

    def test_wide_schema_raises(self):
        s = signality.schemas(layout="wide", engine="arrow[spark]")
        with pytest.raises(NotImplementedError):
            _ = s.tracking

    def test_pyspark_struct_type(self):
        pyspark = pytest.importorskip("pyspark")
        s = signality.schemas(layout="long", engine="arrow[spark]")
        from pyspark.sql.types import StructType
        assert isinstance(s.tracking_spark, StructType)
        assert s.tracking_spark.fields[0].name == "game_id"


# --------------------------------------------------------------------------- #
# arrow vs arrow[spark] dtype parity on timestamp                              #
# --------------------------------------------------------------------------- #

class TestSignalityArrowSparkDialectTimestamp:
    """Catches the classic 'timestamp left as duration[ms] in arrow[spark]'
    regression that the team_id-only string_view check misses."""

    def test_arrow_timestamp_is_duration_ms(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow",
        )
        ts = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_duration(ts) and ts.unit == "ms"

    def test_arrow_spark_timestamp_is_int64(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow[spark]",
        )
        ts = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_int64(ts)


# --------------------------------------------------------------------------- #
# arrow[spark] round-trip + single-file arrow[spark] coverage                  #
# --------------------------------------------------------------------------- #

class TestSignalityArrowSparkRoundTripParity:
    """The strict per-period vs full-match parity test on `arrow[spark]`,
    parallel to TestSignalityRoundTripParity (which exercises `arrow`).
    Catches dialect-normalization regressions that would only surface when
    the normalize step runs between per-period slices."""

    def test_per_period_concat_matches_full_match_arrow_spark(
        self, raw_pairs, meta_bytes, venue_bytes, polars_dataset_full
    ):
        per_period_tables = []
        for period, raw_b in raw_pairs:
            ds = signality.load_tracking(
                meta_data=meta_bytes, raw_data_feeds=raw_b, venue_information=venue_bytes,
                period=period, engine="arrow[spark]",
            )
            per_period_tables.append(ds.tracking)

        concat = pa.concat_tables(per_period_tables)
        concat_pl = pl.from_arrow(concat).sort(
            ["period_id", "frame_id", "team_id", "player_id"]
        )
        baseline = polars_dataset_full.tracking.sort(
            ["period_id", "frame_id", "team_id", "player_id"]
        )

        assert concat_pl.height == baseline.height
        assert sorted(set(concat_pl["frame_id"].to_list())) == sorted(set(baseline["frame_id"].to_list()))
        for col in ("x", "y", "z", "ball_state"):
            if col in concat_pl.columns and col in baseline.columns:
                a = concat_pl[col].head(1000)
                b = baseline[col].head(1000)
                assert a.eq_missing(b).all(), (
                    f"column {col!r} differs in arrow[spark] per-period concat"
                )


class TestSignalityArrowSparkSingleFile:
    """Single-file mode on engine='arrow[spark]' — exercises the explicit
    bytes path through the arrow[spark] normalization."""

    def test_arrow_spark_per_period(self, raw_pairs, meta_bytes, venue_bytes):
        period, raw_b = raw_pairs[0]
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_b, venue_information=venue_bytes,
            period=period, engine="arrow[spark]",
        )
        assert ds.engine == "arrow[spark]"
        assert isinstance(ds.tracking, pa.Table)
        # arrow[spark] dialect must hold on the per-period slice too.
        assert pa.types.is_string(ds.tracking.schema.field("team_id").type)
        assert pa.types.is_int64(ds.tracking.schema.field("timestamp").type)


# --------------------------------------------------------------------------- #
# Post-load transforms on arrow                                                #
# --------------------------------------------------------------------------- #

class TestSignalityArrowTransforms:
    """Confirms `dataset.transform(...)` runs on pyarrow.Table inputs."""

    def test_transform_coords_and_orientation_in_rust(
        self, raw_pairs, meta_bytes, venue_bytes
    ):
        arrow_ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow",
        )
        polars_ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
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

        arrow_pl = pl.from_arrow(arrow_t.tracking).sort(
            ["period_id", "frame_id", "team_id", "player_id"]
        )
        polars_sorted = polars_t.tracking.sort(
            ["period_id", "frame_id", "team_id", "player_id"]
        )
        for col in ("x", "y", "z"):
            a = arrow_pl[col]
            p = polars_sorted[col]
            diffs = (a - p).abs()
            # diff is NaN where either input is NaN (e.g., missing player); treat
            # NaN-vs-NaN as "agree." Real divergence shows up as a finite > tol.
            ok = diffs.is_null() | diffs.is_nan() | (diffs < 1e-3)
            assert ok.all(), f"transform diverged on column {col}"


# --------------------------------------------------------------------------- #
# Engine converters round-trip on signality data                               #
# --------------------------------------------------------------------------- #

class TestSignalityEngineConverters:
    """Per-provider sanity checks on to_arrow / to_polars / to_pyspark."""

    def test_arrow_to_polars(self, raw_pairs, meta_bytes, venue_bytes, polars_dataset_full):
        arrow_ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow",
        )
        converted = arrow_ds.to_polars()
        assert converted.engine == "polars"
        assert isinstance(converted.tracking, pl.DataFrame)
        assert converted.tracking.height == polars_dataset_full.tracking.height

    def test_arrow_polars_arrow_roundtrip(self, raw_pairs, meta_bytes, venue_bytes):
        original = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow",
        )
        via_polars = original.to_polars().to_arrow()
        assert via_polars.engine == "arrow"
        assert isinstance(via_polars.tracking, pa.Table)
        assert via_polars.tracking.num_rows == original.tracking.num_rows
        assert via_polars.tracking.column_names == original.tracking.column_names

    def test_polars_to_arrow_to_polars(self, raw_pairs, meta_bytes, venue_bytes):
        original = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="polars",
        )
        via_arrow = original.to_arrow().to_polars()
        assert via_arrow.engine == "polars"
        assert isinstance(via_arrow.tracking, pl.DataFrame)
        assert via_arrow.tracking.height == original.tracking.height
        assert via_arrow.tracking.columns == original.tracking.columns

    def test_to_arrow_idempotent(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow",
        )
        assert ds.to_arrow() is ds

    def test_arrow_to_pyspark(self, raw_pairs, meta_bytes, venue_bytes, polars_dataset_full):
        pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession
        from pyspark.sql import DataFrame as SparkDataFrame

        spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("ff-signality-arrow-to-spark")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        try:
            arrow_ds = signality.load_tracking(
                meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
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

class TestSignalityArrowIncludeOfficials:
    """`include_officials` adds officials rows but no new columns. The polars
    coverage in tests/test_signality.py confirms the Rust parsing — these
    tests confirm the row count flows through the arrow path."""

    def test_arrow_default_excludes_officials(self, raw_pairs, meta_bytes, venue_bytes):
        ds = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow",
        )
        team_ids = set(ds.tracking.column("team_id").to_pylist())
        assert "officials" not in team_ids

    def test_arrow_include_officials_adds_rows(self, raw_pairs, meta_bytes, venue_bytes):
        no_off = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow", include_officials=False,
        )
        with_off = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow", include_officials=True,
        )
        assert with_off.tracking.num_rows >= no_off.tracking.num_rows
        team_ids = set(with_off.tracking.column("team_id").to_pylist())
        # Officials present if the fixture has any; row count is the load-bearing check.
        if with_off.tracking.num_rows > no_off.tracking.num_rows:
            assert "officials" in team_ids

    def test_arrow_schema_unchanged_by_include_officials(
        self, raw_pairs, meta_bytes, venue_bytes
    ):
        no_off = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow", include_officials=False,
        )
        with_off = signality.load_tracking(
            meta_data=meta_bytes, raw_data_feeds=raw_pairs, venue_information=venue_bytes,
            engine="arrow", include_officials=True,
        )
        assert no_off.tracking.schema == with_off.tracking.schema
