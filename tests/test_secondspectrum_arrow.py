"""Tests for SecondSpectrum engine='arrow' / 'arrow[spark]' paths.

Focused on SecondSpectrum-specific concerns; shared cross-cutting tests
(worker safety, BytesIO acceptance, kloppy lazy-import) live in
tests/test_arrow_output.py and apply equally to every provider on the
arrow path.

Coverage here:
- Return shape: dataset.engine == "arrow", all 5 tables are pyarrow.Table
- Row-count + value parity with engine="polars" baseline
- SecondSpectrum-specific exclude_missing_ball_frames flag works on the
  arrow path
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

from fastforward import secondspectrum
from fastforward._dataset import TrackingDataset
from tests.config import SS_RAW_ANON, SS_META_ANON


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw_bytes():
    with open(SS_RAW_ANON, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_bytes():
    with open(SS_META_ANON, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def polars_dataset(raw_bytes, meta_bytes):
    """Baseline: load with engine='polars' for value comparison."""
    return secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="polars")


# --------------------------------------------------------------------------- #
# Return shape                                                                 #
# --------------------------------------------------------------------------- #

class TestSecondSpectrumArrowShape:

    def test_returns_dataset_with_arrow_tables(self, raw_bytes, meta_bytes):
        ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)
        assert isinstance(ds.metadata, pa.Table)
        assert isinstance(ds.teams, pa.Table)
        assert isinstance(ds.players, pa.Table)
        assert isinstance(ds.periods, pa.Table)

    def test_row_count_matches_polars(self, raw_bytes, meta_bytes, polars_dataset):
        ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert ds.tracking.num_rows == polars_dataset.tracking.height
        assert ds.metadata.num_rows == polars_dataset.metadata.height
        assert ds.teams.num_rows == polars_dataset.teams.height
        assert ds.players.num_rows == polars_dataset.players.height
        assert ds.periods.num_rows == polars_dataset.periods.height


# --------------------------------------------------------------------------- #
# Value parity vs polars baseline                                              #
# --------------------------------------------------------------------------- #

class TestSecondSpectrumArrowValueParity:

    def test_tracking_values_match_polars(self, raw_bytes, meta_bytes, polars_dataset):
        """Column-by-column equality after the uint->int cast on the polars side."""
        arrow_ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
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
# exclude_missing_ball_frames still works on the arrow path                    #
# --------------------------------------------------------------------------- #

class TestSecondSpectrumArrowExcludeMissingBallFrames:
    """The flag plumbs through to the arrow code path and matches polars."""

    def test_polars_arrow_parity_default(self, raw_bytes, meta_bytes):
        # Default exclude_missing_ball_frames=True
        pl_ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="polars")
        arrow_ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height

    def test_polars_arrow_parity_flag_off(self, raw_bytes, meta_bytes):
        # exclude_missing_ball_frames=False should keep more (or equal) rows;
        # the contract we pin here is just polars/arrow agree on row count.
        pl_ds = secondspectrum.load_tracking(
            raw_bytes, meta_bytes, engine="polars", exclude_missing_ball_frames=False,
        )
        arrow_ds = secondspectrum.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", exclude_missing_ball_frames=False,
        )
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height

    def test_flag_off_keeps_at_least_as_many_rows(self, raw_bytes, meta_bytes):
        # Robust to fixture content: flag=False can never DROP rows vs flag=True.
        keep_more = secondspectrum.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", exclude_missing_ball_frames=False,
        )
        keep_default = secondspectrum.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", exclude_missing_ball_frames=True,
        )
        assert keep_more.tracking.num_rows >= keep_default.tracking.num_rows


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestSecondSpectrumArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_loads(self, raw_bytes, meta_bytes, layout):
        ds = secondspectrum.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", layout=layout,
        )
        assert ds.tracking.num_rows > 0

    def test_wide_layout_raises_not_implemented(self, raw_bytes, meta_bytes):
        with pytest.raises(NotImplementedError, match="wide"):
            secondspectrum.load_tracking(
                raw_bytes, meta_bytes, engine="arrow", layout="wide",
            )


# --------------------------------------------------------------------------- #
# Arrow vs arrow[spark] dialect                                                #
# --------------------------------------------------------------------------- #

class TestSecondSpectrumArrowSparkDialect:

    def test_arrow_uses_string_view(self, raw_bytes, meta_bytes):
        ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string_view(team_id_t), (
            f"expected string_view under engine='arrow', got {team_id_t}"
        )

    def test_arrow_spark_uses_string(self, raw_bytes, meta_bytes):
        ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t), (
            f"expected plain string under engine='arrow[spark]', got {team_id_t}"
        )

    def test_arrow_spark_timestamp_is_int64_ms(self, raw_bytes, meta_bytes):
        ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        ts_t = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_int64(ts_t), (
            f"expected int64 under engine='arrow[spark]', got {ts_t}"
        )


# --------------------------------------------------------------------------- #
# Schema factory                                                               #
# --------------------------------------------------------------------------- #

class TestSecondSpectrumArrowSchemas:

    def test_schemas_factory_matches_dataset_schema(self, raw_bytes, meta_bytes):
        ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        s = secondspectrum.schemas(layout="long", engine="arrow[spark]")
        assert s.tracking == ds.tracking.schema
        assert s.metadata == ds.metadata.schema
        assert s.teams == ds.teams.schema
        assert s.players == ds.players.schema
        assert s.periods == ds.periods.schema

    def test_dataset_schemas_property_matches_factory(self, raw_bytes, meta_bytes):
        ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        factory = secondspectrum.schemas(layout="long", engine="arrow[spark]")
        assert ds.schemas.tracking == factory.tracking
        assert ds.schemas.tracking_spark == factory.tracking_spark

    def test_wide_layout_schemas_raises(self):
        s = secondspectrum.schemas(layout="wide", engine="arrow[spark]")
        with pytest.raises(NotImplementedError):
            _ = s.tracking
        with pytest.raises(NotImplementedError):
            _ = s.tracking_spark

    def test_schemas_engine_polars_uses_polars_dialect(self):
        s = secondspectrum.schemas(layout="long", engine="polars")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_schemas_engine_arrow_spark_uses_spark_dialect(self):
        s = secondspectrum.schemas(layout="long", engine="arrow[spark]")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)

    def test_pyspark_struct_type_available(self):
        pyspark = pytest.importorskip("pyspark")
        s = secondspectrum.schemas(layout="long", engine="arrow[spark]")
        from pyspark.sql.types import StructType
        assert isinstance(s.tracking_spark, StructType)
        # First field should be game_id (string) when include_game_id=True (default)
        first = s.tracking_spark.fields[0]
        assert first.name == "game_id"


# --------------------------------------------------------------------------- #
# Phase B additions                                                            #
# --------------------------------------------------------------------------- #

from tests._arrow_helpers import (
    assert_arrow_transform_matches_polars,
    assert_arrow_to_polars_height_match,
    assert_arrow_polars_arrow_roundtrip,
    assert_polars_to_arrow_to_polars,
    assert_to_arrow_idempotent,
    assert_arrow_accepts_bytes_like,
    assert_arrow_rejects_paths,
)


class TestSecondSpectrumArrowTimestampDialect:

    def test_arrow_timestamp_is_duration_ms(self, raw_bytes, meta_bytes):
        ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        ts_t = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_duration(ts_t) and ts_t.unit == "ms"


class TestSecondSpectrumArrowTransforms:

    def test_transform_coords_and_orientation(self, raw_bytes, meta_bytes):
        arrow_ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        polars_ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="polars")
        assert_arrow_transform_matches_polars(arrow_ds, polars_ds)


class TestSecondSpectrumEngineConverters:

    def test_arrow_to_polars(self, raw_bytes, meta_bytes, polars_dataset):
        arrow_ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert_arrow_to_polars_height_match(arrow_ds, polars_dataset)

    def test_arrow_polars_arrow_roundtrip(self, raw_bytes, meta_bytes):
        arrow_ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert_arrow_polars_arrow_roundtrip(arrow_ds)

    def test_polars_to_arrow_to_polars(self, polars_dataset):
        assert_polars_to_arrow_to_polars(polars_dataset)

    def test_to_arrow_idempotent(self, raw_bytes, meta_bytes):
        arrow_ds = secondspectrum.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert_to_arrow_idempotent(arrow_ds)


def _ss_load_arrow(r, m):
    return secondspectrum.load_tracking(r, m, engine="arrow")


class TestSecondSpectrumArrowInputContract:

    def test_accepts_bytes_like_forms(self, raw_bytes, meta_bytes):
        assert_arrow_accepts_bytes_like(_ss_load_arrow, raw_bytes, meta_bytes)

    def test_path_string_rejected_on_arrow(self):
        assert_arrow_rejects_paths(_ss_load_arrow, SS_RAW_ANON, SS_META_ANON)
