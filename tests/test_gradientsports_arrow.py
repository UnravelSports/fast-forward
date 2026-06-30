"""Tests for GradientSports engine='arrow' / 'arrow[spark]' paths.

Focused on GradientSports-specific concerns; shared cross-cutting tests
(worker safety, BytesIO acceptance, kloppy lazy-import) live in
tests/test_arrow_output.py and apply equally to every provider on the
arrow path.

Coverage here:
- Return shape: dataset.engine == "arrow", all 5 tables are pyarrow.Table
- Row-count + value parity with engine="polars" baseline
- GradientSports-specific include_incomplete_frames flag works on the
  arrow path
- 3rd byte buffer (roster_data) plumbs through the framework's
  extra_bytes_params machinery — accepts bytes and resolves FileLike on
  the polars path
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

from fastforward import gradientsports
from fastforward._dataset import TrackingDataset
from tests.config import GS_RAW, GS_META, GS_ROSTER


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw_bytes():
    with open(GS_RAW, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_bytes():
    with open(GS_META, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def roster_bytes():
    with open(GS_ROSTER, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def polars_dataset(raw_bytes, meta_bytes, roster_bytes):
    """Baseline: load with engine='polars' for value comparison."""
    return gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="polars")


# --------------------------------------------------------------------------- #
# Return shape                                                                 #
# --------------------------------------------------------------------------- #

class TestGradientsportsArrowShape:

    def test_returns_dataset_with_arrow_tables(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)
        assert isinstance(ds.metadata, pa.Table)
        assert isinstance(ds.teams, pa.Table)
        assert isinstance(ds.players, pa.Table)
        assert isinstance(ds.periods, pa.Table)

    def test_row_count_matches_polars(self, raw_bytes, meta_bytes, roster_bytes, polars_dataset):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert ds.tracking.num_rows == polars_dataset.tracking.height
        assert ds.metadata.num_rows == polars_dataset.metadata.height
        assert ds.teams.num_rows == polars_dataset.teams.height
        assert ds.players.num_rows == polars_dataset.players.height
        assert ds.periods.num_rows == polars_dataset.periods.height


# --------------------------------------------------------------------------- #
# Value parity vs polars baseline                                              #
# --------------------------------------------------------------------------- #

class TestGradientsportsArrowValueParity:

    def test_tracking_values_match_polars(self, raw_bytes, meta_bytes, roster_bytes, polars_dataset):
        """Column-by-column equality after the uint->int cast on the polars side."""
        arrow_ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
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
# include_incomplete_frames still works on the arrow path                      #
# --------------------------------------------------------------------------- #

class TestGradientsportsArrowIncludeIncompleteFrames:
    """The flag plumbs through to the arrow code path and matches polars."""

    def test_polars_arrow_parity_default(self, raw_bytes, meta_bytes, roster_bytes):
        # Default include_incomplete_frames=False
        pl_ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="polars")
        arrow_ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height

    def test_polars_arrow_parity_flag_on(self, raw_bytes, meta_bytes, roster_bytes):
        # include_incomplete_frames=True should keep more (or equal) rows;
        # the contract we pin here is just polars/arrow agree on row count.
        pl_ds = gradientsports.load_tracking(
            raw_bytes, meta_bytes, roster_bytes, engine="polars", include_incomplete_frames=True,
        )
        arrow_ds = gradientsports.load_tracking(
            raw_bytes, meta_bytes, roster_bytes, engine="arrow", include_incomplete_frames=True,
        )
        assert arrow_ds.tracking.num_rows == pl_ds.tracking.height

    def test_flag_on_keeps_at_least_as_many_rows(self, raw_bytes, meta_bytes, roster_bytes):
        # Robust to fixture content: flag=True can never DROP rows vs flag=False.
        keep_more = gradientsports.load_tracking(
            raw_bytes, meta_bytes, roster_bytes, engine="arrow", include_incomplete_frames=True,
        )
        keep_default = gradientsports.load_tracking(
            raw_bytes, meta_bytes, roster_bytes, engine="arrow", include_incomplete_frames=False,
        )
        assert keep_more.tracking.num_rows >= keep_default.tracking.num_rows


# --------------------------------------------------------------------------- #
# 3rd-buffer plumbing — roster_data via extra_bytes_params                     #
# --------------------------------------------------------------------------- #

class TestGradientsportsRosterDataPlumbing:
    """Confirms the framework's extra_bytes_params hook actually routes
    roster_data correctly on both engines. Bytes input works on the arrow path
    (no kloppy); FileLike (str path) works on the polars path."""

    def test_arrow_accepts_roster_bytes(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert ds.players.num_rows > 0

    def test_polars_accepts_roster_filelike(self):
        # Pass roster_data as a str path, framework should kloppy-resolve it.
        ds = gradientsports.load_tracking(GS_RAW, GS_META, GS_ROSTER, engine="polars")
        assert ds.players.height > 0


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestGradientsportsArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_loads(self, raw_bytes, meta_bytes, roster_bytes, layout):
        ds = gradientsports.load_tracking(
            raw_bytes, meta_bytes, roster_bytes, engine="arrow", layout=layout,
        )
        assert ds.tracking.num_rows > 0

    def test_wide_layout_raises_not_implemented(self, raw_bytes, meta_bytes, roster_bytes):
        with pytest.raises(NotImplementedError, match="wide"):
            gradientsports.load_tracking(
                raw_bytes, meta_bytes, roster_bytes, engine="arrow", layout="wide",
            )


# --------------------------------------------------------------------------- #
# Arrow vs arrow[spark] dialect                                                #
# --------------------------------------------------------------------------- #

class TestGradientsportsArrowSparkDialect:

    def test_arrow_uses_string_view(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert_arrow_engine_uses_string_view(ds)

    def test_arrow_spark_uses_string(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow[spark]")
        assert_arrow_spark_engine_uses_string(ds)

    def test_arrow_spark_timestamp_is_int64_ms(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow[spark]")
        assert_arrow_spark_engine_timestamp_int64(ds)


# --------------------------------------------------------------------------- #
# Schema factory                                                               #
# --------------------------------------------------------------------------- #

class TestGradientsportsArrowSchemas:

    def test_schemas_factory_matches_dataset_schema(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow[spark]")
        s = gradientsports.schemas(layout="long", engine="arrow[spark]")
        assert_schemas_factory_matches_dataset(s, ds)

    def test_dataset_schemas_property_matches_factory(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow[spark]")
        factory = gradientsports.schemas(layout="long", engine="arrow[spark]")
        assert_dataset_schemas_property_matches_factory(ds, factory)

    def test_wide_layout_schemas_raises(self):
        s = gradientsports.schemas(layout="wide", engine="arrow[spark]")
        assert_wide_layout_schemas_raises(s)

    def test_schemas_engine_polars_uses_polars_dialect(self):
        s = gradientsports.schemas(layout="long", engine="polars")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_schemas_engine_arrow_spark_uses_spark_dialect(self):
        s = gradientsports.schemas(layout="long", engine="arrow[spark]")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)

    def test_pyspark_struct_type_available(self):
        s = gradientsports.schemas(layout="long", engine="arrow[spark]")
        assert_pyspark_struct_first_field_is_game_id(s)


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


class TestGradientsportsArrowTimestampDialect:

    def test_arrow_timestamp_is_duration_ms(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert_arrow_engine_timestamp_duration_ms(ds)


class TestGradientsportsArrowTransforms:

    def test_transform_coords_and_orientation(self, raw_bytes, meta_bytes, roster_bytes):
        arrow_ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        polars_ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="polars")
        assert_arrow_transform_matches_polars(arrow_ds, polars_ds)


class TestGradientsportsEngineConverters:

    def test_arrow_to_polars(self, raw_bytes, meta_bytes, roster_bytes, polars_dataset):
        arrow_ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert_arrow_to_polars_height_match(arrow_ds, polars_dataset)

    def test_arrow_polars_arrow_roundtrip(self, raw_bytes, meta_bytes, roster_bytes):
        arrow_ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert_arrow_polars_arrow_roundtrip(arrow_ds)

    def test_polars_to_arrow_to_polars(self, polars_dataset):
        assert_polars_to_arrow_to_polars(polars_dataset)

    def test_to_arrow_idempotent(self, raw_bytes, meta_bytes, roster_bytes):
        arrow_ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert_to_arrow_idempotent(arrow_ds)


def _gs_load_arrow(r, m, rb):
    return gradientsports.load_tracking(r, m, rb, engine="arrow")


class TestGradientsportsArrowInputContract:

    def test_accepts_bytes_like_forms(self, raw_bytes, meta_bytes, roster_bytes):
        assert_arrow_accepts_bytes_like(_gs_load_arrow, raw_bytes, meta_bytes, roster_bytes)

    def test_path_string_rejected_on_arrow(self):
        assert_arrow_rejects_paths(_gs_load_arrow, GS_RAW, GS_META, GS_ROSTER)

    def test_accepts_buffered_reader(self, raw_bytes, meta_bytes, roster_bytes, tmp_path):
        assert_arrow_accepts_buffered_reader(_gs_load_arrow, tmp_path, raw_bytes, meta_bytes, roster_bytes)

    def test_accepts_gzip_stream(self, raw_bytes, meta_bytes, roster_bytes):
        assert_arrow_accepts_gzip_stream(_gs_load_arrow, raw_bytes, meta_bytes, roster_bytes)


class TestGradientsportsArrowIncludeGameId:

    def test_default_includes_game_id(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow")
        assert "game_id" in ds.tracking.column_names

    def test_false_omits_game_id(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow", include_game_id=False)
        assert "game_id" not in ds.tracking.column_names

    def test_str_overrides_game_id(self, raw_bytes, meta_bytes, roster_bytes):
        ds = gradientsports.load_tracking(raw_bytes, meta_bytes, roster_bytes, engine="arrow", include_game_id="custom_123")
        assert set(ds.tracking["game_id"].to_pylist()) == {"custom_123"}
