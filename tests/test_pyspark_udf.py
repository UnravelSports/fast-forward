"""End-to-end Spark mapInArrow integration tests.

These verify the canonical batch-ingest pattern: a Spark DataFrame with one row
per match (tracking_bytes + meta_bytes columns), parsed in parallel via a
mapInArrow UDF using fast-forward's engine='arrow' path inside the UDF.

Also verifies that engine='pyspark' no longer pays the pandas roundtrip.

Gated on pyspark + pyarrow being installed.
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession, Row, DataFrame as SparkDataFrame
from pyspark.sql.types import StructType, StructField, StringType, BinaryType

from fastforward import skillcorner
from tests.config import SC_RAW, SC_META


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def spark():
    sess = (
        SparkSession.builder
        .master("local[2]")
        .appName("fast-forward-udf-test")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    yield sess
    sess.stop()


@pytest.fixture(scope="module")
def raw_bytes():
    with open(SC_RAW, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def meta_bytes():
    with open(SC_META, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def baseline_tracking(raw_bytes, meta_bytes):
    """Polars baseline for value comparison."""
    return skillcorner.load_tracking(
        raw_bytes,
        meta_bytes,
        engine="polars",
        include_ball_owning_player=True,
        include_is_detected=True,
    ).tracking


@pytest.fixture
def matches_input_schema():
    return StructType([
        StructField("match_id", StringType(), False),
        StructField("tracking_bytes", BinaryType(), False),
        StructField("meta_bytes", BinaryType(), False),
    ])


# --------------------------------------------------------------------------- #
# mapInArrow UDF                                                               #
# --------------------------------------------------------------------------- #

def _parse_skillcorner_match_udf(iterator):
    """Spark mapInArrow worker function. Runs per partition.

    Uses engine='arrow[spark]' which returns pre-normalized pyarrow.Tables
    (string, int64) ready for spark.createDataFrame. No manual cast needed.
    Imports happen inside the function so the closure is small and Python
    doesn't pull kloppy onto the worker.
    """
    from fastforward import skillcorner as sc

    for batch in iterator:
        tracking_col = batch.column("tracking_bytes").to_pylist()
        meta_col = batch.column("meta_bytes").to_pylist()
        for raw, meta in zip(tracking_col, meta_col):
            dataset = sc.load_tracking(
                raw,
                meta,
                engine="arrow[spark]",
                layout="long",
                include_game_id=True,
                include_ball_owning_player=True,
                include_is_detected=True,
            )
            for record_batch in dataset.tracking.to_batches():
                yield record_batch


# --------------------------------------------------------------------------- #
# mapInArrow integration                                                       #
# --------------------------------------------------------------------------- #

class TestMapInArrow:

    def test_single_match(self, spark, raw_bytes, meta_bytes, baseline_tracking, matches_input_schema):
        matches_df = spark.createDataFrame(
            [Row(match_id="m1", tracking_bytes=raw_bytes, meta_bytes=meta_bytes)],
            schema=matches_input_schema,
        )
        out_schema = skillcorner.schemas(
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        ).tracking_spark
        result = matches_df.mapInArrow(_parse_skillcorner_match_udf, schema=out_schema)
        assert result.count() == baseline_tracking.height

    def test_three_matches_parallel(self, spark, raw_bytes, meta_bytes, baseline_tracking, matches_input_schema):
        rows = [
            Row(match_id=f"m{i}", tracking_bytes=raw_bytes, meta_bytes=meta_bytes)
            for i in range(3)
        ]
        matches_df = spark.createDataFrame(rows, schema=matches_input_schema)
        out_schema = skillcorner.schemas(
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        ).tracking_spark
        result = matches_df.mapInArrow(_parse_skillcorner_match_udf, schema=out_schema)
        assert result.count() == 3 * baseline_tracking.height

    def test_values_match_baseline(self, spark, raw_bytes, meta_bytes, baseline_tracking, matches_input_schema):
        matches_df = spark.createDataFrame(
            [Row(match_id="m1", tracking_bytes=raw_bytes, meta_bytes=meta_bytes)],
            schema=matches_input_schema,
        )
        out_schema = skillcorner.schemas(
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        ).tracking_spark
        result = matches_df.mapInArrow(_parse_skillcorner_match_udf, schema=out_schema)

        # Sample a handful of frame_id + player_id rows, confirm position match
        spark_rows = (
            result
            .select("frame_id", "player_id", "x", "y", "z")
            .orderBy("frame_id", "player_id")
            .limit(20)
            .collect()
        )
        baseline_lookup = {
            (row["frame_id"], row["player_id"]): (row["x"], row["y"], row["z"])
            for row in (
                baseline_tracking
                .select(["frame_id", "player_id", "x", "y", "z"])
                .sort(["frame_id", "player_id"])
                .head(50)
                .iter_rows(named=True)
            )
        }
        for r in spark_rows:
            key = (r["frame_id"], r["player_id"])
            if key not in baseline_lookup:
                continue
            x, y, z = baseline_lookup[key]
            if x is not None:
                assert abs(r["x"] - x) < 1e-3
                assert abs(r["y"] - y) < 1e-3
                assert abs(r["z"] - z) < 1e-3

    def test_empty_input(self, spark, matches_input_schema):
        empty_df = spark.createDataFrame([], schema=matches_input_schema)
        out_schema = skillcorner.schemas(layout="long").tracking_spark
        result = empty_df.mapInArrow(_parse_skillcorner_match_udf, schema=out_schema)
        assert result.count() == 0
        assert [f.name for f in result.schema.fields] == [f.name for f in out_schema.fields]

    def test_propagates_parse_errors(self, spark, meta_bytes, matches_input_schema):
        matches_df = spark.createDataFrame(
            [Row(match_id="bad", tracking_bytes=b"not json", meta_bytes=meta_bytes)],
            schema=matches_input_schema,
        )
        out_schema = skillcorner.schemas(layout="long").tracking_spark
        result = matches_df.mapInArrow(_parse_skillcorner_match_udf, schema=out_schema)
        with pytest.raises(Exception):
            result.count()


# --------------------------------------------------------------------------- #
# Schema helper                                                                #
# --------------------------------------------------------------------------- #

class TestSparkSchemaHelper:

    def test_create_empty_dataframe(self, spark):
        """The Spark schema must be usable in spark.createDataFrame on an empty DF."""
        s = skillcorner.schemas(
            layout="long",
            include_game_id=True,
            include_ball_owning_player=True,
            include_is_detected=True,
        ).tracking_spark
        df = spark.createDataFrame([], schema=s)
        assert df.count() == 0
        assert len(df.schema.fields) > 0


# --------------------------------------------------------------------------- #
# engine='pyspark' no-pandas-roundtrip guarantee                               #
# --------------------------------------------------------------------------- #

class TestPySparkNoPandasRoundtrip:

    def test_pyspark_engine_does_not_call_pandas(self, spark, raw_bytes, meta_bytes, monkeypatch):
        """engine='pyspark' should NOT instantiate pandas.DataFrame during conversion.

        The old _engine.polars_to_spark went polars -> arrow -> pandas -> spark.
        Post-refactor (load_tracking_arrow -> spark.createDataFrame(arrow_table)),
        pandas.DataFrame must never be touched on the conversion path.
        """
        import pandas

        original_init = pandas.DataFrame.__init__
        calls = []

        def sentinel_init(self, *args, **kwargs):
            calls.append((args, kwargs))
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(pandas.DataFrame, "__init__", sentinel_init)

        dataset = skillcorner.load_tracking(
            raw_bytes,
            meta_bytes,
            engine="pyspark",
            spark_session=spark,
            include_ball_owning_player=True,
            include_is_detected=True,
        )
        assert isinstance(dataset.tracking, SparkDataFrame)
        assert calls == [], (
            f"pandas.DataFrame was instantiated {len(calls)} time(s) "
            "during engine='pyspark' conversion — the pandas roundtrip is back."
        )
