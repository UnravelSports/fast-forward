"""Tests for OptaVision engine='arrow' / 'arrow[spark]' paths.

Focused on OptaVision-specific concerns; shared cross-cutting tests
(worker safety, BytesIO acceptance, kloppy lazy-import) live in
tests/test_arrow_output.py and apply equally to every provider on the
arrow path.

Coverage here:
- Return shape: dataset.engine == "arrow", all 5 tables are pyarrow.Table
- Row-count + value parity with engine="polars" baseline
- OptaVision-specific include_ball_owning_player flag works on the arrow
  path AND adds the ball_owning_player_id column when True
- Layout matrix: long, long_ball, wide rejection
- Arrow vs arrow[spark] dialect difference
- Schema parity: dataset.schemas.tracking matches dataset.tracking.schema
- schemas() factory returns the same shape regardless of how it's reached

Gated on pyarrow being installed.
"""

from __future__ import annotations

import polars as pl
import pytest

pa = pytest.importorskip("pyarrow")

from fastforward import optavision
from fastforward._dataset import TrackingDataset
from tests.config import OV_RAW, OV_META


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw_bytes():
    with open(OV_RAW, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_bytes():
    with open(OV_META, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def polars_dataset(raw_bytes, meta_bytes):
    """Baseline: load with engine='polars' for value comparison."""
    return optavision.load_tracking(raw_bytes, meta_bytes, engine="polars")


# --------------------------------------------------------------------------- #
# Return shape                                                                 #
# --------------------------------------------------------------------------- #

class TestOptavisionArrowShape:

    def test_returns_dataset_with_arrow_tables(self, raw_bytes, meta_bytes):
        ds = optavision.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)
        assert isinstance(ds.metadata, pa.Table)
        assert isinstance(ds.teams, pa.Table)
        assert isinstance(ds.players, pa.Table)
        assert isinstance(ds.periods, pa.Table)

    def test_row_count_matches_polars(self, raw_bytes, meta_bytes, polars_dataset):
        ds = optavision.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert ds.tracking.num_rows == polars_dataset.tracking.height
        assert ds.metadata.num_rows == polars_dataset.metadata.height
        assert ds.teams.num_rows == polars_dataset.teams.height
        assert ds.players.num_rows == polars_dataset.players.height
        assert ds.periods.num_rows == polars_dataset.periods.height


# --------------------------------------------------------------------------- #
# Value parity vs polars baseline                                              #
# --------------------------------------------------------------------------- #

class TestOptavisionArrowValueParity:

    def test_tracking_values_match_polars(self, raw_bytes, meta_bytes, polars_dataset):
        """Column-by-column equality after the uint->int cast on the polars side."""
        arrow_ds = optavision.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        pl_tracking = polars_dataset.tracking
        cast_exprs = []
        for col in pl_tracking.columns:
            dt = pl_tracking[col].dtype
            if dt == pl.UInt8:
                cast_exprs.append(pl.col(col).cast(pl.Int16))
            elif dt == pl.UInt16:
                cast_exprs.append(pl.col(col).cast(pl.Int32))
            elif dt in (pl.UInt32, pl.UInt64):
                cast_exprs.append(pl.col(col).cast(pl.Int64))
            else:
                cast_exprs.append(pl.col(col))
        pl_as_arrow = pl_tracking.select(cast_exprs).to_arrow()
        assert arrow_ds.tracking.column_names == pl_as_arrow.column_names
        for col in arrow_ds.tracking.column_names:
            assert (
                arrow_ds.tracking[col].to_pylist()
                == pl_as_arrow[col].to_pylist()
            ), f"column {col} differs"


# --------------------------------------------------------------------------- #
# include_ball_owning_player matrix (schema-affecting flag)                    #
# --------------------------------------------------------------------------- #

class TestOptavisionArrowIncludeBallOwningPlayer:
    """include_ball_owning_player is a schema-affecting flag — it adds the
    ball_owning_player_id column. Both states must work on the arrow path AND
    schemas() must reflect the choice."""

    def test_flag_true_adds_column(self, raw_bytes, meta_bytes):
        ds = optavision.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", include_ball_owning_player=True,
        )
        assert "ball_owning_player_id" in ds.tracking.column_names

    def test_flag_false_omits_column(self, raw_bytes, meta_bytes):
        ds = optavision.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", include_ball_owning_player=False,
        )
        assert "ball_owning_player_id" not in ds.tracking.column_names

    def test_schema_factory_with_flag_true(self):
        s = optavision.schemas(layout="long", include_ball_owning_player=True)
        assert "ball_owning_player_id" in s.tracking.names

    def test_schema_factory_with_flag_false(self):
        s = optavision.schemas(layout="long", include_ball_owning_player=False)
        assert "ball_owning_player_id" not in s.tracking.names

    def test_polars_arrow_row_count_parity(self, raw_bytes, meta_bytes):
        # Flag doesn't affect rows; both engines should agree regardless.
        pl_ds = optavision.load_tracking(
            raw_bytes, meta_bytes, engine="polars", include_ball_owning_player=True,
        )
        arrow_ds = optavision.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", include_ball_owning_player=True,
        )
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestOptavisionArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_loads(self, raw_bytes, meta_bytes, layout):
        ds = optavision.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", layout=layout,
        )
        assert ds.tracking.num_rows > 0

    def test_wide_layout_raises_not_implemented(self, raw_bytes, meta_bytes):
        with pytest.raises(NotImplementedError, match="wide"):
            optavision.load_tracking(
                raw_bytes, meta_bytes, engine="arrow", layout="wide",
            )


# --------------------------------------------------------------------------- #
# Arrow vs arrow[spark] dialect                                                #
# --------------------------------------------------------------------------- #

class TestOptavisionArrowSparkDialect:

    def test_arrow_uses_string_view(self, raw_bytes, meta_bytes):
        ds = optavision.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string_view(team_id_t), (
            f"expected string_view under engine='arrow', got {team_id_t}"
        )

    def test_arrow_spark_uses_string(self, raw_bytes, meta_bytes):
        ds = optavision.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t), (
            f"expected plain string under engine='arrow[spark]', got {team_id_t}"
        )

    def test_arrow_spark_timestamp_is_int64_ms(self, raw_bytes, meta_bytes):
        ds = optavision.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        ts_t = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_int64(ts_t), (
            f"expected int64 under engine='arrow[spark]', got {ts_t}"
        )


# --------------------------------------------------------------------------- #
# Schema factory                                                               #
# --------------------------------------------------------------------------- #

class TestOptavisionArrowSchemas:

    def test_schemas_factory_matches_dataset_schema(self, raw_bytes, meta_bytes):
        ds = optavision.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        s = optavision.schemas(
            layout="long", include_ball_owning_player=True, engine="arrow[spark]",
        )
        assert s.tracking == ds.tracking.schema
        assert s.metadata == ds.metadata.schema
        assert s.teams == ds.teams.schema
        assert s.players == ds.players.schema
        assert s.periods == ds.periods.schema

    def test_dataset_schemas_property_matches_factory(self, raw_bytes, meta_bytes):
        ds = optavision.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        factory = optavision.schemas(
            layout="long", include_ball_owning_player=True, engine="arrow[spark]",
        )
        assert ds.schemas.tracking == factory.tracking
        assert ds.schemas.tracking_spark == factory.tracking_spark

    def test_wide_layout_schemas_raises(self):
        s = optavision.schemas(layout="wide", engine="arrow[spark]")
        with pytest.raises(NotImplementedError):
            _ = s.tracking
        with pytest.raises(NotImplementedError):
            _ = s.tracking_spark

    def test_schemas_engine_polars_uses_polars_dialect(self):
        s = optavision.schemas(layout="long", engine="polars")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_schemas_engine_arrow_spark_uses_spark_dialect(self):
        s = optavision.schemas(layout="long", engine="arrow[spark]")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)

    def test_pyspark_struct_type_available(self):
        pyspark = pytest.importorskip("pyspark")
        s = optavision.schemas(layout="long", engine="arrow[spark]")
        from pyspark.sql.types import StructType
        assert isinstance(s.tracking_spark, StructType)
        # First field should be game_id (string) when include_game_id=True (default)
        first = s.tracking_spark.fields[0]
        assert first.name == "game_id"
