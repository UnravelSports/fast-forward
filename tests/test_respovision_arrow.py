"""Tests for Respovision engine='arrow' / 'arrow[spark]' paths.

Focused on Respovision-specific concerns; Respovision has a single-buffer
shape so it doesn't go through `_load_tracking_impl` — the arrow branch is
hand-rolled inside the wrapper. The cross-cutting tests in
tests/test_arrow_output.py do NOT exercise this provider (they're scoped
to SkillCorner), so explicit input-contract coverage lives here.

Coverage here:
- Return shape: dataset.engine == "arrow", all 5 tables are pyarrow.Table
- Row-count + value parity with engine="polars" baseline
- Input contract on arrow: bytes / BytesIO / bytearray / memoryview accepted,
  str/Path rejected (covers the hand-rolled _to_bytes path)
- include_joint_angles flag is schema-affecting — both states must work AND
  the schema reflects the choice
- Layout matrix: long, long_ball, wide rejection
- Arrow vs arrow[spark] dialect difference
- Schema parity: dataset.schemas.tracking matches dataset.tracking.schema
- Filename plumbing for game_id derivation on the arrow path

Gated on pyarrow being installed.
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest

pa = pytest.importorskip("pyarrow")

from fastforward import respovision
from fastforward._dataset import TrackingDataset
from tests.config import RV_RAW


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw_bytes():
    with open(RV_RAW, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def rv_filename():
    return Path(RV_RAW).name


@pytest.fixture(scope="module")
def polars_dataset():
    """Baseline: load with engine='polars' for value comparison.

    Goes through the FileLike path so game_id auto-derives from the filename
    on disk; the arrow path needs an explicit `filename=` kwarg for the same
    behavior."""
    return respovision.load_tracking(RV_RAW, engine="polars")


# --------------------------------------------------------------------------- #
# Return shape                                                                 #
# --------------------------------------------------------------------------- #

class TestRespovisionArrowShape:

    def test_returns_dataset_with_arrow_tables(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert isinstance(ds, TrackingDataset)
        assert ds.engine == "arrow"
        assert isinstance(ds.tracking, pa.Table)
        assert isinstance(ds.metadata, pa.Table)
        assert isinstance(ds.teams, pa.Table)
        assert isinstance(ds.players, pa.Table)
        assert isinstance(ds.periods, pa.Table)

    def test_row_count_matches_polars(self, raw_bytes, rv_filename, polars_dataset):
        ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert ds.tracking.num_rows == polars_dataset.tracking.height
        assert ds.metadata.num_rows == polars_dataset.metadata.height
        assert ds.teams.num_rows == polars_dataset.teams.height
        assert ds.players.num_rows == polars_dataset.players.height
        assert ds.periods.num_rows == polars_dataset.periods.height


# --------------------------------------------------------------------------- #
# Value parity vs polars baseline                                              #
# --------------------------------------------------------------------------- #

class TestRespovisionArrowValueParity:

    def test_tracking_values_match_polars(self, raw_bytes, rv_filename, polars_dataset):
        """Column-by-column equality after the uint->int cast on the polars side."""
        arrow_ds = respovision.load_tracking(
            raw_bytes, engine="arrow", filename=rv_filename,
        )
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
# Input contract on the arrow path                                             #
# --------------------------------------------------------------------------- #

class TestRespovisionArrowInputContract:
    """Respovision's arrow path uses the same _to_bytes helper as the shared
    framework. Cross-cutting tests in test_arrow_output.py only exercise
    SkillCorner; explicit coverage for respovision sits here."""

    def test_accepts_bytes(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert ds.tracking.num_rows > 0

    def test_accepts_bytesio(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            io.BytesIO(raw_bytes), engine="arrow", filename=rv_filename,
        )
        assert ds.tracking.num_rows > 0

    def test_accepts_buffered_reader(self, raw_bytes, rv_filename, tmp_path):
        raw_p = tmp_path / "raw.bin"
        raw_p.write_bytes(raw_bytes)
        with open(raw_p, "rb") as rh:
            ds = respovision.load_tracking(rh, engine="arrow", filename=rv_filename)
        assert ds.tracking.num_rows > 0
        assert ds.engine == "arrow"

    def test_accepts_gzip_stream(self, raw_bytes, rv_filename):
        import gzip
        with gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(raw_bytes))) as rh:
            ds = respovision.load_tracking(rh, engine="arrow", filename=rv_filename)
        assert ds.tracking.num_rows > 0
        assert ds.engine == "arrow"

    def test_accepts_bytearray(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            bytearray(raw_bytes), engine="arrow", filename=rv_filename,
        )
        assert ds.tracking.num_rows > 0

    def test_accepts_memoryview(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            memoryview(raw_bytes), engine="arrow", filename=rv_filename,
        )
        assert ds.tracking.num_rows > 0

    def test_rejects_str_path(self):
        with pytest.raises(TypeError, match="bytes or a binary file-like"):
            respovision.load_tracking(RV_RAW, engine="arrow")

    def test_rejects_pathlib(self):
        with pytest.raises(TypeError, match="bytes or a binary file-like"):
            respovision.load_tracking(Path(RV_RAW), engine="arrow")

    def test_rejects_text_io(self):
        with open(RV_RAW, "r") as f:
            with pytest.raises(TypeError, match="bytes"):
                respovision.load_tracking(f, engine="arrow")


# --------------------------------------------------------------------------- #
# include_joint_angles matrix (schema-affecting flag)                          #
# --------------------------------------------------------------------------- #

class TestRespovisionArrowIncludeJointAngles:
    """include_joint_angles adds three columns (head_angle, shoulders_angle,
    hips_angle) on long/long_ball layouts. Both flag states must work AND
    the schema factory must agree with loaded data."""

    def test_flag_true_adds_columns(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            raw_bytes, engine="arrow", filename=rv_filename, include_joint_angles=True,
        )
        for col in ("head_angle", "shoulders_angle", "hips_angle"):
            assert col in ds.tracking.column_names

    def test_flag_false_omits_columns(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            raw_bytes, engine="arrow", filename=rv_filename, include_joint_angles=False,
        )
        for col in ("head_angle", "shoulders_angle", "hips_angle"):
            assert col not in ds.tracking.column_names

    def test_schema_factory_with_flag_true(self):
        s = respovision.schemas(layout="long", include_joint_angles=True)
        for col in ("head_angle", "shoulders_angle", "hips_angle"):
            assert col in s.tracking.names

    def test_schema_factory_with_flag_false(self):
        s = respovision.schemas(layout="long", include_joint_angles=False)
        for col in ("head_angle", "shoulders_angle", "hips_angle"):
            assert col not in s.tracking.names


# --------------------------------------------------------------------------- #
# Layout matrix                                                                #
# --------------------------------------------------------------------------- #

class TestRespovisionArrowLayouts:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_layout_loads(self, raw_bytes, rv_filename, layout):
        ds = respovision.load_tracking(
            raw_bytes, engine="arrow", filename=rv_filename, layout=layout,
        )
        assert ds.tracking.num_rows > 0

    def test_wide_layout_raises_not_implemented(self, raw_bytes, rv_filename):
        with pytest.raises(NotImplementedError, match="wide"):
            respovision.load_tracking(
                raw_bytes, engine="arrow", filename=rv_filename, layout="wide",
            )


# --------------------------------------------------------------------------- #
# Arrow vs arrow[spark] dialect                                                #
# --------------------------------------------------------------------------- #

class TestRespovisionArrowSparkDialect:

    def test_arrow_uses_string_view(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert_arrow_engine_uses_string_view(ds)

    def test_arrow_spark_uses_string(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            raw_bytes, engine="arrow[spark]", filename=rv_filename,
        )
        assert_arrow_spark_engine_uses_string(ds)

    def test_arrow_spark_timestamp_is_int64_ms(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            raw_bytes, engine="arrow[spark]", filename=rv_filename,
        )
        assert_arrow_spark_engine_timestamp_int64(ds)


# --------------------------------------------------------------------------- #
# Schema factory                                                               #
# --------------------------------------------------------------------------- #

class TestRespovisionArrowSchemas:

    def test_schemas_factory_matches_dataset_schema(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            raw_bytes, engine="arrow[spark]", filename=rv_filename,
        )
        s = respovision.schemas(
            layout="long", include_joint_angles=True, engine="arrow[spark]",
        )
        assert_schemas_factory_matches_dataset(s, ds)

    def test_dataset_schemas_property_matches_factory(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(
            raw_bytes, engine="arrow[spark]", filename=rv_filename,
        )
        factory = respovision.schemas(
            layout="long", include_joint_angles=True, engine="arrow[spark]",
        )
        assert_dataset_schemas_property_matches_factory(ds, factory)

    def test_wide_layout_schemas_raises(self):
        s = respovision.schemas(layout="wide", engine="arrow[spark]")
        assert_wide_layout_schemas_raises(s)

    def test_schemas_engine_polars_uses_polars_dialect(self):
        s = respovision.schemas(layout="long", engine="polars")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string_view(team_id_t)

    def test_schemas_engine_arrow_spark_uses_spark_dialect(self):
        s = respovision.schemas(layout="long", engine="arrow[spark]")
        team_id_t = s.tracking.field("team_id").type
        assert pa.types.is_string(team_id_t) and not pa.types.is_string_view(team_id_t)

    def test_pyspark_struct_type_available(self):
        s = respovision.schemas(layout="long", engine="arrow[spark]")
        assert_pyspark_struct_first_field_is_game_id(s)


# --------------------------------------------------------------------------- #
# Phase B additions (InputContract already present in TestRespovisionArrowInputContract) #
# --------------------------------------------------------------------------- #

from tests._arrow_helpers import (
    assert_arrow_transform_matches_polars,
    assert_arrow_to_polars_height_match,
    assert_arrow_polars_arrow_roundtrip,
    assert_polars_to_arrow_to_polars,
    assert_to_arrow_idempotent,
    assert_arrow_engine_uses_string_view,
    assert_arrow_spark_engine_uses_string,
    assert_arrow_engine_timestamp_duration_ms,
    assert_arrow_spark_engine_timestamp_int64,
    assert_schemas_factory_matches_dataset,
    assert_dataset_schemas_property_matches_factory,
    assert_wide_layout_schemas_raises,
    assert_pyspark_struct_first_field_is_game_id,
)


class TestRespovisionArrowTimestampDialect:

    def test_arrow_timestamp_is_duration_ms(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert_arrow_engine_timestamp_duration_ms(ds)


class TestRespovisionArrowTransforms:

    def test_transform_coords_and_orientation(self, raw_bytes, rv_filename):
        arrow_ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        polars_ds = respovision.load_tracking(raw_bytes, engine="polars", filename=rv_filename)
        assert_arrow_transform_matches_polars(arrow_ds, polars_ds)


class TestRespovisionEngineConverters:

    def test_arrow_to_polars(self, raw_bytes, rv_filename, polars_dataset):
        arrow_ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert_arrow_to_polars_height_match(arrow_ds, polars_dataset)

    def test_arrow_polars_arrow_roundtrip(self, raw_bytes, rv_filename):
        arrow_ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert_arrow_polars_arrow_roundtrip(arrow_ds)

    def test_polars_to_arrow_to_polars(self, polars_dataset):
        assert_polars_to_arrow_to_polars(polars_dataset)

    def test_to_arrow_idempotent(self, raw_bytes, rv_filename):
        arrow_ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert_to_arrow_idempotent(arrow_ds)


class TestRespovisionArrowIncludeGameId:

    def test_default_includes_game_id(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename)
        assert "game_id" in ds.tracking.column_names

    def test_false_omits_game_id(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename, include_game_id=False)
        assert "game_id" not in ds.tracking.column_names

    def test_str_overrides_game_id(self, raw_bytes, rv_filename):
        ds = respovision.load_tracking(raw_bytes, engine="arrow", filename=rv_filename, include_game_id="custom_123")
        assert set(ds.tracking["game_id"].to_pylist()) == {"custom_123"}
