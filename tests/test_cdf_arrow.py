"""Tests for CDF engine='arrow' / 'arrow[spark]' paths.

Focused on CDF-specific concerns; shared cross-cutting tests (worker
safety, BytesIO acceptance, kloppy lazy-import) live in
tests/test_arrow_output.py and apply equally to every provider on the
arrow path.

Coverage here:
- Return shape: dataset.engine == "arrow", all 5 tables are pyarrow.Table
- Row-count + value parity with engine="polars" baseline
- CDF-specific exclude_missing_ball_frames flag works on the arrow path
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

from fastforward import cdf
from fastforward._dataset import TrackingDataset
from tests.config import CDF_RAW, CDF_META


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw_bytes():
    with open(CDF_RAW, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_bytes():
    with open(CDF_META, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def polars_dataset(raw_bytes, meta_bytes):
    """Baseline: load with engine='polars' for value comparison."""
    return cdf.load_tracking(raw_bytes, meta_bytes, engine="polars")


# --------------------------------------------------------------------------- #
# Return shape                                                                 #
# --------------------------------------------------------------------------- #

class TestCdfArrowShape:

    def test_returns_dataset_with_arrow_tables(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)
        assert isinstance(ds.metadata, pa.Table)
        assert isinstance(ds.teams, pa.Table)
        assert isinstance(ds.players, pa.Table)
        assert isinstance(ds.periods, pa.Table)

    def test_row_count_matches_polars(self, raw_bytes, meta_bytes, polars_dataset):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert ds.tracking.num_rows == polars_dataset.tracking.height
        assert ds.metadata.num_rows == polars_dataset.metadata.height
        assert ds.teams.num_rows == polars_dataset.teams.height
        assert ds.players.num_rows == polars_dataset.players.height
        assert ds.periods.num_rows == polars_dataset.periods.height


# --------------------------------------------------------------------------- #
# Value parity vs polars baseline                                              #
# --------------------------------------------------------------------------- #

class TestCdfArrowValueParity:

    def test_tracking_values_match_polars(self, raw_bytes, meta_bytes, polars_dataset):
        """Column-by-column equality after the uint->int cast on the polars side."""
        arrow_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
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

class TestCdfArrowExcludeMissingBallFrames:
    """The flag plumbs through to the arrow code path and matches polars."""

    def test_polars_arrow_parity_default(self, raw_bytes, meta_bytes):
        # Default exclude_missing_ball_frames=True
        pl_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="polars")
        arrow_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height

    def test_polars_arrow_parity_flag_off(self, raw_bytes, meta_bytes):
        # exclude_missing_ball_frames=False should keep more (or equal) rows;
        # the contract we pin here is just polars/arrow agree on row count.
        pl_ds = cdf.load_tracking(
            raw_bytes, meta_bytes, engine="polars", exclude_missing_ball_frames=False,
        )
        arrow_ds = cdf.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", exclude_missing_ball_frames=False,
        )
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height

    def test_flag_off_keeps_at_least_as_many_rows(self, raw_bytes, meta_bytes):
        # Robust to fixture content: flag=False can never DROP rows vs flag=True.
        keep_more = cdf.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", exclude_missing_ball_frames=False,
        )
        keep_default = cdf.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", exclude_missing_ball_frames=True,
        )
        assert keep_more.tracking.num_rows >= keep_default.tracking.num_rows


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestCdfArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_loads(self, raw_bytes, meta_bytes, layout):
        ds = cdf.load_tracking(
            raw_bytes, meta_bytes, engine="arrow", layout=layout,
        )
        assert ds.tracking.num_rows > 0

    def test_wide_layout_raises_not_implemented(self, raw_bytes, meta_bytes):
        with pytest.raises(NotImplementedError, match="wide"):
            cdf.load_tracking(
                raw_bytes, meta_bytes, engine="arrow", layout="wide",
            )


# --------------------------------------------------------------------------- #
# Arrow vs arrow[spark] dialect                                                #
# --------------------------------------------------------------------------- #

class TestCdfArrowSparkDialect:

    def test_arrow_uses_string_view(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert_arrow_engine_uses_string_view(ds)

    def test_arrow_spark_uses_string(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        assert_arrow_spark_engine_uses_string(ds)

    def test_arrow_spark_timestamp_is_int64_ms(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        assert_arrow_spark_engine_timestamp_int64(ds)


class TestCdfArrowTimestampDialect:

    def test_arrow_timestamp_is_duration_ms(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert_arrow_engine_timestamp_duration_ms(ds)


# --------------------------------------------------------------------------- #
# Schema factory                                                               #
# --------------------------------------------------------------------------- #

class TestCdfArrowSchemas:

    def test_schemas_factory_matches_dataset_schema(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        s = cdf.schemas(layout="long", engine="arrow[spark]")
        assert_schemas_factory_matches_dataset(s, ds)

    def test_dataset_schemas_property_matches_factory(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow[spark]")
        factory = cdf.schemas(layout="long", engine="arrow[spark]")
        assert_dataset_schemas_property_matches_factory(ds, factory)

    def test_wide_layout_schemas_raises(self):
        s = cdf.schemas(layout="wide", engine="arrow[spark]")
        assert_wide_layout_schemas_raises(s)

    def test_schemas_engine_polars_uses_polars_dialect(self):
        s = cdf.schemas(layout="long", engine="polars")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_schemas_engine_arrow_spark_uses_spark_dialect(self):
        s = cdf.schemas(layout="long", engine="arrow[spark]")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)

    def test_pyspark_struct_type_available(self):
        s = cdf.schemas(layout="long", engine="arrow[spark]")
        assert_pyspark_struct_first_field_is_game_id(s)


# --------------------------------------------------------------------------- #
# Post-load transforms                                                         #
# --------------------------------------------------------------------------- #

from tests._arrow_helpers import (
    assert_arrow_transform_matches_polars,
    assert_arrow_to_polars_height_match,
    assert_arrow_polars_arrow_roundtrip,
    assert_polars_to_arrow_to_polars,
    assert_to_arrow_idempotent,
    assert_arrow_accepts_bytes_like,
    assert_arrow_rejects_paths,
    assert_arrow_accepts_buffered_reader,
    assert_arrow_accepts_gzip_stream,
    assert_arrow_engine_uses_string_view,
    assert_arrow_spark_engine_uses_string,
    assert_arrow_engine_timestamp_duration_ms,
    assert_arrow_spark_engine_timestamp_int64,
    assert_schemas_factory_matches_dataset,
    assert_dataset_schemas_property_matches_factory,
    assert_wide_layout_schemas_raises,
    assert_pyspark_struct_first_field_is_game_id,
)
from tests.config import CDF_RAW, CDF_META


class TestCdfArrowTransforms:

    def test_transform_coords_and_orientation(self, raw_bytes, meta_bytes):
        arrow_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        polars_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="polars")
        assert_arrow_transform_matches_polars(arrow_ds, polars_ds)


# --------------------------------------------------------------------------- #
# Engine-converter round-trips                                                 #
# --------------------------------------------------------------------------- #

class TestCdfEngineConverters:

    def test_arrow_to_polars(self, raw_bytes, meta_bytes, polars_dataset):
        arrow_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert_arrow_to_polars_height_match(arrow_ds, polars_dataset)

    def test_arrow_polars_arrow_roundtrip(self, raw_bytes, meta_bytes):
        arrow_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert_arrow_polars_arrow_roundtrip(arrow_ds)

    def test_polars_to_arrow_to_polars(self, polars_dataset):
        assert_polars_to_arrow_to_polars(polars_dataset)

    def test_to_arrow_idempotent(self, raw_bytes, meta_bytes):
        arrow_ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert_to_arrow_idempotent(arrow_ds)


# --------------------------------------------------------------------------- #
# Input contract — bytes-only on arrow                                         #
# --------------------------------------------------------------------------- #

def _load_arrow(r, m):
    return cdf.load_tracking(r, m, engine="arrow")


class TestCdfArrowInputContract:

    def test_accepts_bytes_like_forms(self, raw_bytes, meta_bytes):
        assert_arrow_accepts_bytes_like(_load_arrow, raw_bytes, meta_bytes)

    def test_path_string_rejected_on_arrow(self):
        assert_arrow_rejects_paths(_load_arrow, CDF_RAW, CDF_META)

    def test_accepts_buffered_reader(self, raw_bytes, meta_bytes, tmp_path):
        assert_arrow_accepts_buffered_reader(_load_arrow, tmp_path, raw_bytes, meta_bytes)

    def test_accepts_gzip_stream(self, raw_bytes, meta_bytes):
        assert_arrow_accepts_gzip_stream(_load_arrow, raw_bytes, meta_bytes)


class TestCdfArrowIncludeGameId:

    def test_default_includes_game_id(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow")
        assert "game_id" in ds.tracking.column_names

    def test_false_omits_game_id(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow", include_game_id=False)
        assert "game_id" not in ds.tracking.column_names

    def test_str_overrides_game_id(self, raw_bytes, meta_bytes):
        ds = cdf.load_tracking(raw_bytes, meta_bytes, engine="arrow", include_game_id="custom_123")
        assert set(ds.tracking["game_id"].to_pylist()) == {"custom_123"}
