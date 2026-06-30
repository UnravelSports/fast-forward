"""Tests for StatsPerform engine='arrow' / 'arrow[spark]' paths.

Focused on StatsPerform-specific concerns; shared cross-cutting tests
(worker safety, BytesIO acceptance, kloppy lazy-import) live in
tests/test_arrow_output.py and apply equally to every provider on the
arrow path.

Coverage here:
- Return shape: dataset.engine == "arrow", all 5 tables are pyarrow.Table
- Row-count + value parity with engine="polars" baseline
- Dual-format matrix (MA1 JSON + MA1 XML metadata both work on arrow path)
- include_officials flag plumbs through (rows-only, so polars/arrow agree)
- Layout matrix: long, long_ball, wide rejection
- Arrow vs arrow[spark] dialect difference
- Schema parity: dataset.schemas.tracking matches dataset.tracking.schema

Gated on pyarrow being installed.
"""

from __future__ import annotations

import polars as pl
import pytest

pa = pytest.importorskip("pyarrow")

from fastforward import statsperform
from fastforward._dataset import TrackingDataset
from tests.config import STP_RAW_MA25, STP_META_JSON, STP_META_XML


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw_bytes():
    with open(STP_RAW_MA25, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_json_bytes():
    with open(STP_META_JSON, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_xml_bytes():
    with open(STP_META_XML, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def polars_dataset(raw_bytes, meta_json_bytes):
    """Baseline: load with engine='polars' for value comparison."""
    return statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="polars")


# --------------------------------------------------------------------------- #
# Return shape                                                                 #
# --------------------------------------------------------------------------- #

class TestStatsperformArrowShape:

    def test_returns_dataset_with_arrow_tables(self, raw_bytes, meta_json_bytes):
        ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)
        assert isinstance(ds.metadata, pa.Table)
        assert isinstance(ds.teams, pa.Table)
        assert isinstance(ds.players, pa.Table)
        assert isinstance(ds.periods, pa.Table)

    def test_row_count_matches_polars(self, raw_bytes, meta_json_bytes, polars_dataset):
        ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        assert ds.tracking.num_rows == polars_dataset.tracking.height
        assert ds.metadata.num_rows == polars_dataset.metadata.height
        assert ds.teams.num_rows == polars_dataset.teams.height
        assert ds.players.num_rows == polars_dataset.players.height
        assert ds.periods.num_rows == polars_dataset.periods.height


# --------------------------------------------------------------------------- #
# Value parity vs polars baseline                                              #
# --------------------------------------------------------------------------- #

class TestStatsperformArrowValueParity:

    def test_tracking_values_match_polars(self, raw_bytes, meta_json_bytes, polars_dataset):
        """Column-by-column equality after the uint->int cast on the polars side."""
        arrow_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
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
# MA1 JSON vs MA1 XML metadata format autodetection on arrow path              #
# --------------------------------------------------------------------------- #

class TestStatsperformArrowMetadataFormatMatrix:
    """MA1 metadata autodetection (JSON vs XML) sits beneath the arrow path.
    Confirm the arrow path follows the autodetection correctly on each combo."""

    def test_json_metadata(self, raw_bytes, meta_json_bytes):
        pl_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="polars")
        arrow_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height
        assert arrow_ds.players.num_rows == pl_ds.players.height

    def test_xml_metadata(self, raw_bytes, meta_xml_bytes):
        pl_ds = statsperform.load_tracking(raw_bytes, meta_xml_bytes, engine="polars")
        arrow_ds = statsperform.load_tracking(raw_bytes, meta_xml_bytes, engine="arrow")
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height
        assert arrow_ds.players.num_rows == pl_ds.players.height


# --------------------------------------------------------------------------- #
# include_officials still works on the arrow path                              #
# --------------------------------------------------------------------------- #

class TestStatsperformArrowIncludeOfficials:
    """include_officials adds officials rows to player_df. polars and arrow
    must agree on the count for both states of the flag."""

    def test_polars_arrow_parity_default(self, raw_bytes, meta_json_bytes):
        pl_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="polars")
        arrow_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        assert arrow_ds.players.num_rows == pl_ds.players.height

    def test_polars_arrow_parity_flag_on(self, raw_bytes, meta_json_bytes):
        pl_ds = statsperform.load_tracking(
            raw_bytes, meta_json_bytes, engine="polars", include_officials=True,
        )
        arrow_ds = statsperform.load_tracking(
            raw_bytes, meta_json_bytes, engine="arrow", include_officials=True,
        )
        assert arrow_ds.players.num_rows == pl_ds.players.height

    def test_flag_on_keeps_at_least_as_many_player_rows(self, raw_bytes, meta_json_bytes):
        keep_more = statsperform.load_tracking(
            raw_bytes, meta_json_bytes, engine="arrow", include_officials=True,
        )
        keep_default = statsperform.load_tracking(
            raw_bytes, meta_json_bytes, engine="arrow", include_officials=False,
        )
        assert keep_more.players.num_rows >= keep_default.players.num_rows


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestStatsperformArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_loads(self, raw_bytes, meta_json_bytes, layout):
        ds = statsperform.load_tracking(
            raw_bytes, meta_json_bytes, engine="arrow", layout=layout,
        )
        assert ds.tracking.num_rows > 0

    def test_wide_layout_raises_not_implemented(self, raw_bytes, meta_json_bytes):
        with pytest.raises(NotImplementedError, match="wide"):
            statsperform.load_tracking(
                raw_bytes, meta_json_bytes, engine="arrow", layout="wide",
            )


# --------------------------------------------------------------------------- #
# Arrow vs arrow[spark] dialect                                                #
# --------------------------------------------------------------------------- #

class TestStatsperformArrowSparkDialect:

    def test_arrow_uses_string_view(self, raw_bytes, meta_json_bytes):
        ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string_view(team_id_t), (
            f"expected string_view under engine='arrow', got {team_id_t}"
        )

    def test_arrow_spark_uses_string(self, raw_bytes, meta_json_bytes):
        ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow[spark]")
        team_id_t = ds.tracking.schema.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t), (
            f"expected plain string under engine='arrow[spark]', got {team_id_t}"
        )

    def test_arrow_spark_timestamp_is_int64_ms(self, raw_bytes, meta_json_bytes):
        ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow[spark]")
        ts_t = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_int64(ts_t), (
            f"expected int64 under engine='arrow[spark]', got {ts_t}"
        )


# --------------------------------------------------------------------------- #
# Schema factory                                                               #
# --------------------------------------------------------------------------- #

class TestStatsperformArrowSchemas:

    def test_schemas_factory_matches_dataset_schema(self, raw_bytes, meta_json_bytes):
        ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow[spark]")
        s = statsperform.schemas(layout="long", engine="arrow[spark]")
        assert s.tracking == ds.tracking.schema
        assert s.metadata == ds.metadata.schema
        assert s.teams == ds.teams.schema
        assert s.players == ds.players.schema
        assert s.periods == ds.periods.schema

    def test_dataset_schemas_property_matches_factory(self, raw_bytes, meta_json_bytes):
        ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow[spark]")
        factory = statsperform.schemas(layout="long", engine="arrow[spark]")
        assert ds.schemas.tracking == factory.tracking
        assert ds.schemas.tracking_spark == factory.tracking_spark

    def test_wide_layout_schemas_raises(self):
        s = statsperform.schemas(layout="wide", engine="arrow[spark]")
        with pytest.raises(NotImplementedError):
            _ = s.tracking
        with pytest.raises(NotImplementedError):
            _ = s.tracking_spark

    def test_schemas_engine_polars_uses_polars_dialect(self):
        s = statsperform.schemas(layout="long", engine="polars")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_schemas_engine_arrow_spark_uses_spark_dialect(self):
        s = statsperform.schemas(layout="long", engine="arrow[spark]")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)

    def test_pyspark_struct_type_available(self):
        pyspark = pytest.importorskip("pyspark")
        s = statsperform.schemas(layout="long", engine="arrow[spark]")
        from pyspark.sql.types import StructType
        assert isinstance(s.tracking_spark, StructType)
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


class TestStatsperformArrowTimestampDialect:

    def test_arrow_timestamp_is_duration_ms(self, raw_bytes, meta_json_bytes):
        ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        ts_t = ds.tracking.schema.field("timestamp").type
        assert pa.types.is_duration(ts_t) and ts_t.unit == "ms"


class TestStatsperformArrowTransforms:

    def test_transform_coords_and_orientation(self, raw_bytes, meta_json_bytes):
        arrow_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        polars_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="polars")
        assert_arrow_transform_matches_polars(arrow_ds, polars_ds)


class TestStatsperformEngineConverters:

    def test_arrow_to_polars(self, raw_bytes, meta_json_bytes, polars_dataset):
        arrow_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        assert_arrow_to_polars_height_match(arrow_ds, polars_dataset)

    def test_arrow_polars_arrow_roundtrip(self, raw_bytes, meta_json_bytes):
        arrow_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        assert_arrow_polars_arrow_roundtrip(arrow_ds)

    def test_polars_to_arrow_to_polars(self, polars_dataset):
        assert_polars_to_arrow_to_polars(polars_dataset)

    def test_to_arrow_idempotent(self, raw_bytes, meta_json_bytes):
        arrow_ds = statsperform.load_tracking(raw_bytes, meta_json_bytes, engine="arrow")
        assert_to_arrow_idempotent(arrow_ds)


def _stp_load_arrow(r, m):
    return statsperform.load_tracking(r, m, engine="arrow")


class TestStatsperformArrowInputContract:

    def test_accepts_bytes_like_forms(self, raw_bytes, meta_json_bytes):
        assert_arrow_accepts_bytes_like(_stp_load_arrow, raw_bytes, meta_json_bytes)

    def test_path_string_rejected_on_arrow(self):
        assert_arrow_rejects_paths(_stp_load_arrow, STP_RAW_MA25, STP_META_JSON)
