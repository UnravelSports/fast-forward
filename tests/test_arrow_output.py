"""Tests for engine='arrow' — Rust-owned Arrow output path.

These tests pin down the contract that:
- ``engine="arrow"`` returns a ``TrackingDataset`` whose 5 tables are
  ``pyarrow.Table`` instances.
- Bytes-only input is enforced (paths/BytesIO raise TypeError).
- The worker-safe contract: no kloppy import on this code path.
- The uint→int cast happens in Rust for arrow output (PySpark compat) but
  does NOT change ``engine="polars"`` dtypes (regression guard).
- Post-load ``.transform(...)`` still runs in Rust on arrow data.

Gated on pyarrow being installed.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

pa = pytest.importorskip("pyarrow")

from fastforward import skillcorner
from fastforward._dataset import TrackingDataset
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


@pytest.fixture(scope="module")
def polars_dataset(raw_bytes, meta_bytes):
    """Baseline: load with engine='polars' for comparison."""
    return skillcorner.load_tracking(
        raw_bytes,
        meta_bytes,
        engine="polars",
        include_ball_owning_player=True,
        include_is_detected=True,
    )


# --------------------------------------------------------------------------- #
# Return shape                                                                 #
# --------------------------------------------------------------------------- #

class TestArrowOutputShape:

    def test_returns_dataset_with_arrow_tables(self, raw_bytes, meta_bytes):
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert isinstance(dataset, TrackingDataset)
        assert dataset.engine == "arrow"
        assert isinstance(dataset.tracking, pa.Table)
        assert isinstance(dataset.metadata, pa.Table)
        assert isinstance(dataset.teams, pa.Table)
        assert isinstance(dataset.players, pa.Table)
        assert isinstance(dataset.periods, pa.Table)

    def test_row_count_matches_polars(self, raw_bytes, meta_bytes, polars_dataset):
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert dataset.tracking.num_rows == polars_dataset.tracking.height
        assert dataset.metadata.num_rows == polars_dataset.metadata.height
        assert dataset.teams.num_rows == polars_dataset.teams.height
        assert dataset.players.num_rows == polars_dataset.players.height
        assert dataset.periods.num_rows == polars_dataset.periods.height


# --------------------------------------------------------------------------- #
# Value equality vs. polars baseline                                           #
# --------------------------------------------------------------------------- #

class TestArrowValueParity:

    def test_values_match_polars(self, raw_bytes, meta_bytes, polars_dataset):
        """Column-by-column equality after the uint→int cast on the polars side."""
        arrow_dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )

        # Convert polars baseline to pyarrow (after explicit uint→int cast on
        # baseline so it matches the rust-side cast applied for arrow engine)
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
        pl_cast = pl_tracking.select(cast_exprs)
        pl_as_arrow = pl_cast.to_arrow()

        assert arrow_dataset.tracking.column_names == pl_as_arrow.column_names
        for col in arrow_dataset.tracking.column_names:
            assert (
                arrow_dataset.tracking[col].to_pylist()
                == pl_as_arrow[col].to_pylist()
            ), f"column {col} differs"


# --------------------------------------------------------------------------- #
# Dtype contract                                                               #
# --------------------------------------------------------------------------- #

class TestArrowDtypes:

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_arrow_dtypes_are_signed(self, raw_bytes, meta_bytes, layout):
        """All UInt columns in the polars schema must be signed in arrow output."""
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            layout=layout,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        for field in dataset.tracking.schema:
            assert not pa.types.is_unsigned_integer(field.type), (
                f"column {field.name} has unsigned dtype {field.type} under engine='arrow'"
            )

    def test_polars_dtypes_unchanged(self, raw_bytes, meta_bytes):
        """Regression guard: engine='polars' (default) still returns UInt dtypes."""
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="polars",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        # frame_id is UInt32 today; this is the single most likely silent-change
        # surface. If this dtype shifts, downstream user joins/asserts break.
        assert dataset.tracking["frame_id"].dtype == pl.UInt32


# --------------------------------------------------------------------------- #
# Input-type contract (bytes-only for engine='arrow')                          #
# --------------------------------------------------------------------------- #

class TestArrowInputContract:

    def test_rejects_str_path_input(self, meta_bytes):
        with pytest.raises(TypeError, match="bytes"):
            skillcorner.load_tracking(SC_RAW, meta_bytes, engine="arrow")

    def test_rejects_pathlib_input(self, meta_bytes):
        with pytest.raises(TypeError, match="bytes"):
            skillcorner.load_tracking(Path(SC_RAW), meta_bytes, engine="arrow")

    def test_accepts_bytesio_input(self, raw_bytes, meta_bytes):
        """io.BytesIO is pure stdlib and accepted on the arrow path —
        we read it ourselves via .read(). No kloppy / FileLike resolution.
        """
        dataset = skillcorner.load_tracking(
            io.BytesIO(raw_bytes), io.BytesIO(meta_bytes),
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert dataset.tracking.num_rows > 0

    def test_bytesio_equivalent_to_bytes(self, raw_bytes, meta_bytes):
        """BytesIO input must produce identical tracking values to bytes input."""
        ds_bytes = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        ds_io = skillcorner.load_tracking(
            io.BytesIO(raw_bytes), io.BytesIO(meta_bytes),
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert ds_bytes.tracking.num_rows == ds_io.tracking.num_rows
        assert ds_bytes.tracking.column_names == ds_io.tracking.column_names
        for col in ds_bytes.tracking.column_names:
            assert ds_bytes.tracking[col].to_pylist() == ds_io.tracking[col].to_pylist(), (
                f"column {col} differs between bytes and BytesIO input"
            )

    def test_rejects_text_io(self, meta_bytes):
        """Text-mode file objects return str, not bytes — must reject cleanly."""
        # StringIO returns str on .read()
        with pytest.raises(TypeError, match=r"(bytes|stream returned)"):
            skillcorner.load_tracking(
                io.StringIO("not bytes"), meta_bytes, engine="arrow",
            )

    def test_accepts_bytearray(self, raw_bytes, meta_bytes):
        dataset = skillcorner.load_tracking(
            bytearray(raw_bytes),
            bytearray(meta_bytes),
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert dataset.tracking.num_rows > 0

    def test_accepts_memoryview(self, raw_bytes, meta_bytes):
        """memoryview comes naturally out of Arrow buffer columns in workers."""
        dataset = skillcorner.load_tracking(
            memoryview(raw_bytes),
            memoryview(meta_bytes),
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert dataset.tracking.num_rows > 0

    def test_empty_bytes_raises(self, meta_bytes):
        with pytest.raises(ValueError):
            skillcorner.load_tracking(b"", meta_bytes, engine="arrow")
        with pytest.raises(ValueError):
            skillcorner.load_tracking(b"x", b"", engine="arrow")

    def test_arrow_and_spark_session_mutually_exclusive(self, raw_bytes, meta_bytes):
        """Passing both engine='arrow' and spark_session=... is undefined; fail loud."""
        pyspark = pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local[1]").appName("ff-test").getOrCreate()
        try:
            with pytest.raises(TypeError, match="arrow"):
                skillcorner.load_tracking(
                    raw_bytes, meta_bytes,
                    engine="arrow",
                    spark_session=spark,
                )
        finally:
            spark.stop()


# --------------------------------------------------------------------------- #
# Worker safety: no kloppy import on the arrow code path                       #
# --------------------------------------------------------------------------- #

class TestArrowWorkerSafety:

    def test_does_not_import_kloppy(self, raw_bytes, meta_bytes):
        """Run in a fresh subprocess: kloppy must not appear in sys.modules
        before or after a load_tracking(engine='arrow') call.

        This is the contract that lets fast-forward be called from inside a
        Spark mapInArrow worker without dragging kloppy onto every executor.
        """
        script = f"""
import sys
import pickle
import io

# Sanity: kloppy not yet imported
assert 'kloppy' not in sys.modules, 'kloppy was somehow pre-imported'

# Now import fast-forward and call the arrow engine
from fastforward import skillcorner

with open({SC_RAW!r}, 'rb') as f:
    raw = f.read()
with open({SC_META!r}, 'rb') as f:
    meta = f.read()

ds = skillcorner.load_tracking(
    raw, meta,
    engine='arrow',
    include_ball_owning_player=True,
    include_is_detected=True,
)

# Critical: after the load completes, kloppy must STILL not be in sys.modules.
# If this fails, some import inside the arrow code path is dragging kloppy in.
assert 'kloppy' not in sys.modules, (
    f"kloppy was imported during engine='arrow' load. "
    f"Imported kloppy submodules: {{[m for m in sys.modules if m.startswith('kloppy')]}}"
)
print('OK')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"subprocess failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        assert "OK" in result.stdout

    @pytest.mark.skipif(
        os.environ.get("FASTFORWARD_TEST_NO_KLOPPY") != "1",
        reason="Run with FASTFORWARD_TEST_NO_KLOPPY=1 in an env where kloppy is uninstalled.",
    )
    def test_works_without_kloppy_installed(self, raw_bytes, meta_bytes):
        """Run locally with `pip uninstall -y kloppy && FASTFORWARD_TEST_NO_KLOPPY=1 pytest -k no_kloppy`.

        CI runs this in a dedicated job (`worker-safety`) that uninstalls
        kloppy before pytest. The subprocess test above catches the module-load
        leak; this catches a runtime leak inside the arrow code path.
        """
        # If kloppy is uninstalled and the arrow path tried to import it, this
        # call would raise ImportError. The success of this call IS the assertion.
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert dataset.tracking.num_rows > 0


# --------------------------------------------------------------------------- #
# Layouts                                                                      #
# --------------------------------------------------------------------------- #

class TestArrowLayouts:

    def test_layout_long_ball_works(self, raw_bytes, meta_bytes):
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            layout="long_ball",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert dataset.tracking.num_rows > 0
        assert "ball_x" in dataset.tracking.column_names
        assert "ball_y" in dataset.tracking.column_names
        assert "ball_z" in dataset.tracking.column_names

    def test_layout_wide_raises(self, raw_bytes, meta_bytes):
        """Wide has per-game schema (player IDs in column names) — incompatible
        with the static-schema contract that mapInArrow + engine='arrow' rely on.
        """
        with pytest.raises(NotImplementedError, match="wide"):
            skillcorner.load_tracking(
                raw_bytes,
                meta_bytes,
                engine="arrow",
                layout="wide",
                include_ball_owning_player=True,
                include_is_detected=True,
            )


# --------------------------------------------------------------------------- #
# Include-flag matrix                                                          #
# --------------------------------------------------------------------------- #

class TestArrowIncludeFlags:

    @pytest.mark.parametrize("include_game_id", [True, False])
    @pytest.mark.parametrize("include_ball_owning_player", [True, False])
    @pytest.mark.parametrize("include_is_detected", [True, False])
    def test_include_flag_matrix(
        self,
        raw_bytes,
        meta_bytes,
        include_game_id,
        include_ball_owning_player,
        include_is_detected,
    ):
        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_game_id=include_game_id,
            include_ball_owning_player=include_ball_owning_player,
            include_is_detected=include_is_detected,
        )
        cols = set(dataset.tracking.column_names)
        assert ("game_id" in cols) == bool(include_game_id)
        assert ("ball_owning_player_id" in cols) == include_ball_owning_player
        assert ("is_detected" in cols) == include_is_detected


# --------------------------------------------------------------------------- #
# Post-load transforms — run in Rust on arrow data                             #
# --------------------------------------------------------------------------- #

class TestArrowTransforms:

    def test_transform_in_rust(self, raw_bytes, meta_bytes):
        """Load engine='arrow', call .transform(...), assert equality with the
        polars-engine baseline transform. Also confirms the returned dataset
        stays on engine='arrow' (no Python-Polars detour leaked through).
        """
        arrow_dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        polars_dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="polars",
            include_ball_owning_player=True,
            include_is_detected=True,
        )

        arrow_t = arrow_dataset.transform(
            to_coordinates="opta",
            to_orientation="static_away_home",
        )
        polars_t = polars_dataset.transform(
            to_coordinates="opta",
            to_orientation="static_away_home",
        )

        assert arrow_t.engine == "arrow"
        assert isinstance(arrow_t.tracking, pa.Table)

        # Compare x/y/z columns (the transform changes only coordinates+orientation)
        for col in ("x", "y", "z"):
            arrow_vals = arrow_t.tracking[col].to_pylist()
            polars_vals = polars_t.tracking[col].to_list()
            assert len(arrow_vals) == len(polars_vals)
            for a, p in zip(arrow_vals, polars_vals):
                if a is None or p is None:
                    assert a is None and p is None
                else:
                    assert abs(a - p) < 1e-3, f"{col} differs: arrow={a}, polars={p}"


# --------------------------------------------------------------------------- #
# Engine roundtrip converters                                                  #
# --------------------------------------------------------------------------- #

class TestEngineConverters:

    def test_arrow_to_polars_roundtrip(self, raw_bytes, meta_bytes, polars_dataset):
        """dataset.to_polars() on arrow engine -> tracking equals fresh polars load."""
        arrow_dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        converted = arrow_dataset.to_polars()
        assert converted.engine == "polars"
        assert isinstance(converted.tracking, pl.DataFrame)
        assert converted.tracking.height == polars_dataset.tracking.height

    def test_arrow_polars_arrow_roundtrip(self, raw_bytes, meta_bytes):
        """Tier 1 contract: load(engine='arrow') -> .to_polars() -> .to_arrow()
        produces an arrow dataset whose tracking values equal the original.
        """
        ds_original = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        ds_via_polars = ds_original.to_polars().to_arrow()

        assert ds_via_polars.engine == "arrow"
        assert isinstance(ds_via_polars.tracking, pa.Table)
        assert ds_via_polars.tracking.num_rows == ds_original.tracking.num_rows
        assert ds_via_polars.tracking.column_names == ds_original.tracking.column_names
        for col in ds_original.tracking.column_names:
            assert (
                ds_via_polars.tracking[col].to_pylist()
                == ds_original.tracking[col].to_pylist()
            ), f"column {col} differs after arrow → polars → arrow roundtrip"

    def test_polars_to_arrow_to_polars_roundtrip(self, raw_bytes, meta_bytes):
        """The other direction: load(engine='polars') -> .to_arrow() -> .to_polars()."""
        ds_original = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="polars",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        ds_via_arrow = ds_original.to_arrow().to_polars()

        assert ds_via_arrow.engine == "polars"
        assert isinstance(ds_via_arrow.tracking, pl.DataFrame)
        assert ds_via_arrow.tracking.height == ds_original.tracking.height
        assert ds_via_arrow.tracking.columns == ds_original.tracking.columns

    def test_to_arrow_idempotent(self, raw_bytes, meta_bytes):
        """Calling .to_arrow() on an arrow dataset returns self unchanged."""
        ds = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert ds.to_arrow() is ds

    def test_arrow_to_pyspark_roundtrip(self, raw_bytes, meta_bytes, polars_dataset):
        """dataset.to_pyspark() on arrow engine -> tracking equals fresh pyspark load."""
        pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession
        from pyspark.sql import DataFrame as SparkDataFrame

        spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("ff-arrow-to-spark")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        try:
            arrow_dataset = skillcorner.load_tracking(
                raw_bytes,
                meta_bytes,
                engine="arrow",
                include_ball_owning_player=True,
                include_is_detected=True,
            )
            converted = arrow_dataset.to_pyspark(spark)
            assert converted.engine == "pyspark"
            assert isinstance(converted.tracking, SparkDataFrame)
            assert converted.tracking.count() == polars_dataset.tracking.height
        finally:
            spark.stop()


# --------------------------------------------------------------------------- #
# engine="arrow[spark]" — pre-normalized variant                              #
# --------------------------------------------------------------------------- #

class TestArrowSparkDialect:
    """`engine="arrow[spark]"` returns pyarrow.Tables already normalized for
    Spark consumption — `string` (not `string_view`), `int64` (not
    `duration[ms]`). Spark UDFs can yield batches directly without a manual
    cast.
    """

    def test_returns_normalized_types(self, raw_bytes, meta_bytes):
        ds = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow[spark]",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        # game_id is string (not string_view)
        assert ds.tracking.schema.field("game_id").type == pa.string(), (
            f"game_id type: {ds.tracking.schema.field('game_id').type}"
        )
        # timestamp is int64 (not duration[ms])
        assert ds.tracking.schema.field("timestamp").type == pa.int64(), (
            f"timestamp type: {ds.tracking.schema.field('timestamp').type}"
        )
        # No string_view, no duration anywhere
        for field in ds.tracking.schema:
            assert not pa.types.is_string_view(field.type), (
                f"{field.name} is string_view"
            )
            assert not pa.types.is_duration(field.type), (
                f"{field.name} is duration"
            )

    def test_engine_property(self, raw_bytes, meta_bytes):
        ds = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow[spark]",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert ds.engine == "arrow[spark]"

    def test_arrow_keeps_polars_style(self, raw_bytes, meta_bytes):
        """Sanity guard: engine='arrow' still emits string_view + duration."""
        ds = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert pa.types.is_string_view(ds.tracking.schema.field("game_id").type)
        assert pa.types.is_duration(ds.tracking.schema.field("timestamp").type)

    def test_arrow_and_arrow_spark_same_values(self, raw_bytes, meta_bytes):
        """Only dtypes differ between arrow and arrow[spark]; values match."""
        ds_arrow = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        ds_spark = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow[spark]",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert ds_arrow.tracking.num_rows == ds_spark.tracking.num_rows
        assert ds_arrow.tracking.column_names == ds_spark.tracking.column_names
        for col in ds_arrow.tracking.column_names:
            # Both .to_pylist() conversions yield identical Python values
            # regardless of underlying Arrow representation (string_view → str,
            # duration[ms] → timedelta vs int64 → int — the int is the ms count).
            arr_values = ds_arrow.tracking[col].to_pylist()
            sp_values = ds_spark.tracking[col].to_pylist()
            if col == "timestamp":
                # duration[ms] → datetime.timedelta; int64 → int (ms).
                # Compare ms values.
                arr_as_ms = [
                    int(v.total_seconds() * 1000) if v is not None else None
                    for v in arr_values
                ]
                assert arr_as_ms == sp_values
            else:
                assert arr_values == sp_values, f"column {col} differs"

    def test_schemas_match_data_arrow_spark(self, raw_bytes, meta_bytes):
        """dataset.schemas.tracking must match dataset.tracking.schema, including
        dtypes (both should be normalized when engine='arrow[spark]').
        """
        ds = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow[spark]",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        for field in ds.tracking.schema:
            helper_field = ds.schemas.tracking.field(field.name)
            assert field.type == helper_field.type, (
                f"column {field.name}: data type {field.type} != "
                f"schema helper type {helper_field.type}"
            )

    def test_schemas_match_data_arrow(self, raw_bytes, meta_bytes):
        """Same parity check for engine='arrow' (Polars-style)."""
        ds = skillcorner.load_tracking(
            raw_bytes, meta_bytes,
            engine="arrow",
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        for field in ds.tracking.schema:
            helper_field = ds.schemas.tracking.field(field.name)
            assert field.type == helper_field.type, (
                f"column {field.name}: data type {field.type} != "
                f"schema helper type {helper_field.type}"
            )

    def test_arrow_spark_feeds_spark_directly(self, raw_bytes, meta_bytes):
        """The whole point: yield arrow[spark] batches into spark.createDataFrame
        without any manual cast. If string_view or duration leaks through,
        Spark raises UNSUPPORTED_*ARROWTYPE.
        """
        pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession
        spark = (SparkSession.builder.master("local[1]")
                 .appName("ff-arrow-spark-direct")
                 .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                 .getOrCreate())
        try:
            ds = skillcorner.load_tracking(
                raw_bytes, meta_bytes,
                engine="arrow[spark]",
                include_ball_owning_player=True,
                include_is_detected=True,
            )
            # No normalize_arrow_for_spark call — direct.
            sdf = spark.createDataFrame(ds.tracking)
            assert sdf.count() == ds.tracking.num_rows
        finally:
            spark.stop()


# --------------------------------------------------------------------------- #
# Engine validation                                                            #
# --------------------------------------------------------------------------- #

class TestEngineValidation:
    """validate_engine rejects unknown values; the provider surface propagates the same error."""

    def test_unknown_engine_raises(self):
        from fastforward._engine import validate_engine
        with pytest.raises(ValueError, match="Invalid engine"):
            validate_engine("bogus")

    def test_unknown_engine_on_load_tracking_raises(self, raw_bytes, meta_bytes):
        with pytest.raises(ValueError, match="Invalid engine"):
            skillcorner.load_tracking(
                raw_bytes, meta_bytes, engine="bogus",  # type: ignore[arg-type]
                include_ball_owning_player=False, include_is_detected=False,
            )
