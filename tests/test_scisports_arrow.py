"""Tests for SciSports engine='arrow' / 'arrow[spark]' paths.

Coverage matches the per-provider arrow matrix from the new-provider-checklist:
- Return shape: dataset.engine == "arrow", all 5 tables are pyarrow.Table
- Row-count + value parity with engine="polars" baseline
- Layout matrix: long, long_ball, wide rejection
- Dialect: arrow vs arrow[spark] (string_view vs string, duration[ms] vs int64)
- Schema parity: dataset.schemas.tracking matches dataset.tracking.schema
- include_game_id matrix (the only schema-relevant flag on SciSports)
- Post-load transforms run against arrow tables
- Engine converters: to_arrow / to_polars / to_pyspark
- kloppy-free contract: arrow engine rejects FileLike

Gated on pyarrow being installed.
"""

from __future__ import annotations

import io

import polars as pl
import pytest

pa = pytest.importorskip("pyarrow")

from fastforward import scisports
from fastforward._dataset import TrackingDataset
from tests.config import SCI_META_XML, SCI_RAW_TXT


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw_bytes():
    with open(SCI_RAW_TXT, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_bytes():
    with open(SCI_META_XML, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def polars_dataset(raw_bytes, meta_bytes):
    return scisports.load_tracking(raw_bytes, meta_bytes, engine="polars")


# --------------------------------------------------------------------------- #
# Return shape                                                                 #
# --------------------------------------------------------------------------- #

class TestSciSportsArrowShape:

    def test_returns_dataset_with_arrow_tables(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)
        assert isinstance(ds.metadata, pa.Table)
        assert isinstance(ds.teams, pa.Table)
        assert isinstance(ds.players, pa.Table)
        assert isinstance(ds.periods, pa.Table)

    def test_row_count_matches_polars(self, raw_bytes, meta_bytes, polars_dataset):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert ds.tracking.num_rows == polars_dataset.tracking.height
        assert ds.metadata.num_rows == polars_dataset.metadata.height
        assert ds.teams.num_rows == polars_dataset.teams.height
        assert ds.players.num_rows == polars_dataset.players.height
        assert ds.periods.num_rows == polars_dataset.periods.height


# --------------------------------------------------------------------------- #
# Input contract — bytes/bytearray/memoryview/BytesIO accepted on arrow,       #
# FileLike rejected (kloppy-free contract)                                     #
# --------------------------------------------------------------------------- #

class TestSciSportsArrowInputContract:

    def test_accepts_bytes(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert ds.tracking.num_rows > 0

    def test_accepts_bytearray(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(
            bytearray(raw_bytes), bytearray(meta_bytes), engine="arrow"
        )
        assert ds.tracking.num_rows > 0

    def test_accepts_memoryview(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(
            memoryview(raw_bytes), memoryview(meta_bytes), engine="arrow"
        )
        assert ds.tracking.num_rows > 0

    def test_accepts_bytesio(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(
            io.BytesIO(raw_bytes), io.BytesIO(meta_bytes), engine="arrow"
        )
        assert ds.tracking.num_rows > 0

    def test_path_string_rejected_on_arrow(self):
        with pytest.raises(TypeError):
            scisports.load_tracking(
                SCI_RAW_TXT,  # plain path string
                SCI_META_XML,
                engine="arrow",
            )


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestSciSportsArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout(self, raw_bytes, meta_bytes, layout):
        ds = scisports.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", layout=layout
        )
        assert ds.tracking.num_rows > 0

    def test_wide_layout_rejected(self, raw_bytes, meta_bytes):
        with pytest.raises(NotImplementedError, match="wide"):
            scisports.load_tracking(
                raw_bytes, meta_bytes, engine="arrow", layout="wide"
            )


# --------------------------------------------------------------------------- #
# Dialect: arrow vs arrow[spark]                                               #
# --------------------------------------------------------------------------- #

class TestSciSportsArrowSparkDialect:

    def test_arrow_uses_string_view(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_arrow_spark_uses_string(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)

    def test_arrow_timestamp_is_duration_ms(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        ts = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_duration(ts) and ts.unit == "ms"

    def test_arrow_spark_timestamp_is_int64(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        ts = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_int64(ts)


# --------------------------------------------------------------------------- #
# Schema factory parity                                                        #
# --------------------------------------------------------------------------- #

class TestSciSportsArrowSchemas:

    def test_schemas_factory_matches_dataset(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        s = scisports.schemas(layout="long", engine="arrow[spark]")
        assert s.tracking == ds.tracking.schema

    def test_dataset_schemas_property_matches_factory(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        factory = scisports.schemas(layout="long", engine="arrow[spark]")
        assert ds.schemas.tracking == factory.tracking
        assert ds.schemas.tracking_spark == factory.tracking_spark

    def test_wide_layout_schemas_raises(self):
        s = scisports.schemas(layout="wide", engine="arrow[spark]")
        with pytest.raises(NotImplementedError):
            _ = s.tracking

    def test_pyspark_struct_type(self):
        pytest.importorskip("pyspark")
        s = scisports.schemas(layout="long", engine="arrow[spark]")
        from pyspark.sql.types import StructType
        assert isinstance(s.tracking_spark, StructType)
        assert s.tracking_spark.fields[0].name == "game_id"


# --------------------------------------------------------------------------- #
# include_game_id matrix                                                       #
# --------------------------------------------------------------------------- #

class TestSciSportsArrowIncludeGameId:

    def test_default_includes_game_id(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert "game_id" in ds.tracking.column_names

    def test_false_omits_game_id(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", include_game_id=False
        )
        assert "game_id" not in ds.tracking.column_names

    def test_str_overrides_game_id(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", include_game_id="custom_123"
        )
        assert set(ds.tracking["game_id"].to_pylist()) == {"custom_123"}


# --------------------------------------------------------------------------- #
# Post-load transforms — runs against pyarrow.Table on the arrow engine       #
# --------------------------------------------------------------------------- #

class TestSciSportsArrowTransforms:

    def test_transform_coords_and_orientation(self, raw_bytes, meta_bytes):
        arrow_ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        polars_ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="polars")

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
        for col in ("x", "y"):
            a = arrow_pl[col]
            p = polars_sorted[col]
            diffs = (a - p).abs()
            ok = diffs.is_null() | diffs.is_nan() | (diffs < 1e-3)
            assert ok.all(), f"transform diverged on column {col}"


# --------------------------------------------------------------------------- #
# Engine converters round-trip                                                 #
# --------------------------------------------------------------------------- #

class TestSciSportsEngineConverters:

    def test_arrow_to_polars(self, raw_bytes, meta_bytes, polars_dataset):
        arrow_ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        converted = arrow_ds.to_polars()
        assert converted.engine == "polars"
        assert isinstance(converted.tracking, pl.DataFrame)
        assert converted.tracking.height == polars_dataset.tracking.height

    def test_arrow_polars_arrow_roundtrip(self, raw_bytes, meta_bytes):
        original = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        via_polars = original.to_polars().to_arrow()
        assert via_polars.engine == "arrow"
        assert via_polars.tracking.num_rows == original.tracking.num_rows
        assert via_polars.tracking.column_names == original.tracking.column_names

    def test_polars_to_arrow_to_polars(self, raw_bytes, meta_bytes):
        original = scisports.load_tracking(raw_bytes, meta_bytes, engine="polars")
        via_arrow = original.to_arrow().to_polars()
        assert via_arrow.engine == "polars"
        assert via_arrow.tracking.height == original.tracking.height
        assert via_arrow.tracking.columns == original.tracking.columns

    def test_to_arrow_idempotent(self, raw_bytes, meta_bytes):
        ds = scisports.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert ds.to_arrow() is ds
