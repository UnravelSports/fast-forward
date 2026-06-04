"""Tests for the schemas() namespace — Rust-derived schemas for all 5 tables.

The contract:
- Each provider exposes ``schemas(**kwargs) -> Schemas`` returning a namespace
  with 10 lazy properties: ``{tracking,metadata,teams,players,periods}`` (arrow)
  + ``{tracking,metadata,teams,players,periods}_spark`` (PySpark StructType).
- Schemas are derived from the same Rust constants the row builders use —
  asserted by parity tests against loaded data.
- The same ``Schemas`` is reachable via ``dataset.schemas`` after a load,
  with the load's kwargs implicitly bound. Engine doesn't matter
  (works for polars, arrow, pyspark).

Gated on pyarrow being installed.
"""

from __future__ import annotations

import pytest

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


# --------------------------------------------------------------------------- #
# Namespace shape                                                              #
# --------------------------------------------------------------------------- #

class TestSchemasNamespace:

    def test_returns_namespace_with_10_properties(self):
        schemas = skillcorner.schemas(layout="long")
        # Arrow schemas
        assert isinstance(schemas.tracking, pa.Schema)
        assert isinstance(schemas.metadata, pa.Schema)
        assert isinstance(schemas.teams, pa.Schema)
        assert isinstance(schemas.players, pa.Schema)
        assert isinstance(schemas.periods, pa.Schema)
        # Spark schemas (gated on pyspark)
        pyspark = pytest.importorskip("pyspark")
        from pyspark.sql.types import StructType
        assert isinstance(schemas.tracking_spark, StructType)
        assert isinstance(schemas.metadata_spark, StructType)
        assert isinstance(schemas.teams_spark, StructType)
        assert isinstance(schemas.players_spark, StructType)
        assert isinstance(schemas.periods_spark, StructType)

    def test_does_not_load_data(self):
        """schemas(...) should be cheap — no tracking/meta bytes required."""
        import time
        t0 = time.perf_counter()
        s = skillcorner.schemas(
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        # Touch each arrow property at least once to force materialization
        _ = s.tracking, s.metadata, s.teams, s.players, s.periods
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"schemas() too slow: {elapsed:.3f}s"

    def test_caching(self):
        """Repeated access returns the same object (lazy cache)."""
        s = skillcorner.schemas(layout="long")
        assert s.tracking is s.tracking
        assert s.metadata is s.metadata


# --------------------------------------------------------------------------- #
# Tracking schema parity against loaded arrow data                             #
# --------------------------------------------------------------------------- #

class TestTrackingSchemaParity:

    @pytest.mark.parametrize("include_game_id", [True, False])
    @pytest.mark.parametrize("include_ball_owning_player", [True, False])
    @pytest.mark.parametrize("include_is_detected", [True, False])
    def test_long_schema_matches_loaded_arrow_table(
        self,
        raw_bytes,
        meta_bytes,
        include_game_id,
        include_ball_owning_player,
        include_is_detected,
    ):
        schemas = skillcorner.schemas(
            layout="long",
            include_game_id=include_game_id,
            include_ball_owning_player=include_ball_owning_player,
            include_is_detected=include_is_detected,
        )
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            layout="long",
            include_game_id=include_game_id,
            include_ball_owning_player=include_ball_owning_player,
            include_is_detected=include_is_detected,
        )
        assert schemas.tracking.names == dataset.tracking.schema.names
        for name in schemas.tracking.names:
            assert schemas.tracking.field(name).type == dataset.tracking.schema.field(name).type, (
                f"column {name}: schema says {schemas.tracking.field(name).type}, "
                f"loaded says {dataset.tracking.schema.field(name).type}"
            )

    def test_long_ball_schema_matches_loaded_arrow_table(self, raw_bytes, meta_bytes):
        schemas = skillcorner.schemas(
            layout="long_ball",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            layout="long_ball",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert schemas.tracking.names == dataset.tracking.schema.names

    def test_wide_tracking_raises(self):
        """Wide layout has no static schema; metadata/teams/players/periods still work."""
        schemas = skillcorner.schemas(layout="wide")
        with pytest.raises(NotImplementedError, match="wide"):
            _ = schemas.tracking
        # But the other 4 must still work — they're layout-independent.
        assert isinstance(schemas.metadata, pa.Schema)
        assert isinstance(schemas.teams, pa.Schema)
        assert isinstance(schemas.players, pa.Schema)
        assert isinstance(schemas.periods, pa.Schema)


# --------------------------------------------------------------------------- #
# Non-tracking schemas parity against loaded data                              #
# --------------------------------------------------------------------------- #

class TestNonTrackingSchemasParity:

    def test_metadata_schema_matches_loaded(self, raw_bytes, meta_bytes):
        schemas = skillcorner.schemas(layout="long")
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert schemas.metadata.names == dataset.metadata.schema.names

    def test_teams_schema_matches_loaded(self, raw_bytes, meta_bytes):
        schemas = skillcorner.schemas(layout="long", include_game_id=True)
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert schemas.teams.names == dataset.teams.schema.names

    def test_teams_schema_no_game_id(self, raw_bytes, meta_bytes):
        schemas = skillcorner.schemas(layout="long", include_game_id=False)
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_game_id=False,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert schemas.teams.names == dataset.teams.schema.names
        assert "game_id" not in schemas.teams.names

    def test_players_schema_matches_loaded(self, raw_bytes, meta_bytes):
        schemas = skillcorner.schemas(layout="long", include_game_id=True)
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert schemas.players.names == dataset.players.schema.names

    def test_periods_schema_matches_loaded(self, raw_bytes, meta_bytes):
        schemas = skillcorner.schemas(layout="long", include_game_id=True)
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert schemas.periods.names == dataset.periods.schema.names


# --------------------------------------------------------------------------- #
# Spark schema mapping                                                         #
# --------------------------------------------------------------------------- #

class TestSchemasEngineAlias:
    """The schemas() factory accepts the same 4 engine values as load_tracking.
    'polars' and 'arrow' both produce Polars-style arrow schemas; 'pyspark'
    and 'arrow[spark]' both produce Spark-compat arrow schemas.
    """

    @pytest.mark.parametrize("engine_value", ["polars", "arrow"])
    def test_polars_style_engines_use_polars_dialect(self, engine_value):
        s = skillcorner.schemas(layout="long", engine=engine_value)
        # game_id is string_view (Polars-style) for both polars and arrow engines
        assert pa.types.is_string_view(s.tracking.field("game_id").type)

    @pytest.mark.parametrize("engine_value", ["pyspark", "arrow[spark]"])
    def test_spark_style_engines_use_spark_dialect(self, engine_value):
        s = skillcorner.schemas(layout="long", engine=engine_value)
        # game_id is plain string (Spark-compat) for both pyspark and arrow[spark]
        assert s.tracking.field("game_id").type == pa.string()
        # timestamp is int64, not duration
        assert s.tracking.field("timestamp").type == pa.int64()

    def test_default_engine_is_polars_style(self):
        s_default = skillcorner.schemas(layout="long")
        s_explicit = skillcorner.schemas(layout="long", engine="polars")
        assert s_default.tracking.names == s_explicit.tracking.names
        # Both should match Polars-style
        assert pa.types.is_string_view(s_default.tracking.field("game_id").type)


class TestSparkSchemaMapping:

    def test_spark_schema_field_mapping(self):
        """Each Arrow field maps to the expected Spark type."""
        pytest.importorskip("pyspark")
        from pyspark.sql.types import (
            IntegerType, LongType, FloatType, DoubleType, StringType, BooleanType,
        )
        schemas = skillcorner.schemas(
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        arrow_schema = schemas.tracking
        spark_schema = schemas.tracking_spark

        # Same field names, same order
        arrow_names = arrow_schema.names
        spark_names = [f.name for f in spark_schema.fields]
        assert arrow_names == spark_names

        # Spot-check a few critical type mappings — full mapping is the Spark
        # adapter's responsibility, but we verify the obvious ones.
        for f in spark_schema.fields:
            af = arrow_schema.field(f.name)
            if pa.types.is_string(af.type):
                assert isinstance(f.dataType, StringType)
            elif pa.types.is_boolean(af.type):
                assert isinstance(f.dataType, BooleanType)
            elif pa.types.is_float32(af.type):
                assert isinstance(f.dataType, FloatType)
            elif pa.types.is_float64(af.type):
                assert isinstance(f.dataType, DoubleType)
            elif pa.types.is_int32(af.type):
                assert isinstance(f.dataType, IntegerType)
            elif pa.types.is_int64(af.type):
                assert isinstance(f.dataType, LongType)


# --------------------------------------------------------------------------- #
# Dataset.schemas property                                                     #
# --------------------------------------------------------------------------- #

class TestDatasetSchemasProperty:

    def test_property_matches_factory(self, raw_bytes, meta_bytes):
        """dataset.schemas == provider.schemas(**load_kwargs)."""
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        factory = skillcorner.schemas(
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert dataset.schemas.tracking.names == factory.tracking.names
        assert dataset.schemas.metadata.names == factory.metadata.names

    def test_property_works_for_all_engines(self, raw_bytes, meta_bytes):
        """Same StructType regardless of engine the dataset was loaded with."""
        kwargs = dict(
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        polars_ds = skillcorner.load_tracking(raw_bytes, meta_bytes, engine="polars", **kwargs)
        arrow_ds = skillcorner.load_tracking(raw_bytes, meta_bytes, engine="arrow", **kwargs)
        assert polars_ds.schemas.tracking.names == arrow_ds.schemas.tracking.names
