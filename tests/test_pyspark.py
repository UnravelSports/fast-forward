"""Tests for PySpark engine support.

These tests mirror the secondspectrum tests but verify PySpark DataFrame output.
Tests are skipped if PySpark is not installed.
"""

import pytest

# Skip all tests if PySpark is not installed
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame

from fastforward import secondspectrum
from fastforward._dataset import TrackingDataset
from tests.config import (
    SS_RAW_ANON as RAW_DATA_PATH,
    SS_META_ANON as META_DATA_PATH,
)


@pytest.fixture(scope="module")
def spark():
    """Create SparkSession for tests."""
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("fast-forward-test")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )


class TestPySparkEngine:
    """Tests for engine='pyspark' parameter."""

    def test_returns_spark_dataframes(self, spark):
        """Test that engine='pyspark' returns PySpark DataFrames for all properties."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="pyspark",
            spark_session=spark,
        )

        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, SparkDataFrame)
        assert isinstance(dataset.metadata, SparkDataFrame)
        assert isinstance(dataset.teams, SparkDataFrame)
        assert isinstance(dataset.players, SparkDataFrame)
        assert isinstance(dataset.periods, SparkDataFrame)

    def test_engine_property(self, spark):
        """Test that engine property returns 'pyspark'."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="pyspark",
            spark_session=spark,
        )
        assert dataset.engine == "pyspark"

    def test_default_engine_is_polars(self):
        """Test that default engine is 'polars'."""
        import polars as pl

        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
        )
        assert dataset.engine == "polars"
        assert isinstance(dataset.tracking, pl.DataFrame)


class TestPySparkMetadata:
    """Tests for metadata DataFrame with PySpark engine."""

    @pytest.fixture
    def dataset(self, spark):
        """Load dataset with PySpark engine."""
        return secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="pyspark",
            spark_session=spark,
        )

    def test_single_row(self, dataset):
        """Test that metadata contains exactly one row."""
        assert dataset.metadata.count() == 1

    def test_schema_columns(self, dataset):
        """Test that metadata has expected columns."""
        expected_columns = {
            "provider",
            "game_id",
            "game_date",
            "home_team",
            "home_team_id",
            "away_team",
            "away_team_id",
            "pitch_length",
            "pitch_width",
            "fps",
            "coordinate_system",
            "orientation",
        }
        assert set(dataset.metadata.columns) == expected_columns

    def test_provider_value(self, dataset):
        """Test provider is 'secondspectrum'."""
        row = dataset.metadata.first()
        assert row["provider"] == "secondspectrum"

    def test_coordinate_system_value(self, dataset):
        """Test coordinate_system is 'cdf'."""
        row = dataset.metadata.first()
        assert row["coordinate_system"] == "cdf"

    def test_orientation_value(self, dataset):
        """Test orientation is 'static_home_away'."""
        row = dataset.metadata.first()
        assert row["orientation"] == "static_home_away"

    def test_team_names(self, dataset):
        """Test team names are extracted correctly."""
        row = dataset.metadata.first()
        assert row["home_team"] == "HOME"
        assert row["away_team"] == "AWAY"


class TestPySparkTeams:
    """Tests for teams DataFrame with PySpark engine."""

    @pytest.fixture
    def dataset(self, spark):
        """Load dataset with PySpark engine."""
        return secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="pyspark",
            spark_session=spark,
        )

    def test_two_rows(self, dataset):
        """Test that teams contains exactly two rows."""
        assert dataset.teams.count() == 2

    def test_schema_columns(self, dataset):
        """Test that teams has expected columns."""
        expected_columns = {"game_id", "team_id", "name", "ground"}
        assert set(dataset.teams.columns) == expected_columns

    def test_grounds(self, dataset):
        """Test that teams contains home and away."""
        grounds = {row["ground"] for row in dataset.teams.collect()}
        assert grounds == {"home", "away"}


class TestPySparkPlayers:
    """Tests for players DataFrame with PySpark engine."""

    @pytest.fixture
    def dataset(self, spark):
        """Load dataset with PySpark engine."""
        return secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="pyspark",
            spark_session=spark,
        )

    def test_schema_columns(self, dataset):
        """Test that players has expected columns."""
        expected_columns = {
            "game_id", "team_id", "player_id", "name",
            "first_name", "last_name", "jersey_number",
            "position", "is_starter"
        }
        assert set(dataset.players.columns) == expected_columns

    def test_has_players(self, dataset):
        """Test that players contains expected number of players."""
        # Expected: 20 home + 20 away = 40 players
        assert dataset.players.count() == 40


class TestPySparkTrackingLong:
    """Tests for tracking DataFrame with 'long' layout and PySpark engine."""

    @pytest.fixture
    def dataset(self, spark):
        """Load dataset with PySpark engine and long layout."""
        return secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            layout="long",
            engine="pyspark",
            spark_session=spark,
        )

    def test_schema_columns(self, dataset):
        """Test that tracking has expected columns."""
        expected_columns = {
            "game_id",
            "frame_id",
            "period_id",
            "timestamp",
            "ball_state",
            "ball_owning_team_id",
            "team_id",
            "player_id",
            "x",
            "y",
            "z",
        }
        assert set(dataset.tracking.columns) == expected_columns

    def test_row_count(self, dataset):
        """Test that tracking has expected row count (same as Polars)."""
        assert dataset.tracking.count() == 4554

    def test_has_ball_rows(self, dataset):
        """Test that long format includes ball as separate rows."""
        ball_count = dataset.tracking.filter("team_id = 'ball'").count()
        assert ball_count == 198

    def test_has_multiple_periods(self, dataset):
        """Test that data includes two periods."""
        period_count = dataset.tracking.select("period_id").distinct().count()
        assert period_count == 2


class TestPySparkTrackingLongBall:
    """Tests for tracking DataFrame with 'long_ball' layout and PySpark engine."""

    @pytest.fixture
    def dataset(self, spark):
        """Load dataset with PySpark engine and long_ball layout."""
        return secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            layout="long_ball",
            engine="pyspark",
            spark_session=spark,
        )

    def test_schema_columns(self, dataset):
        """Test that tracking has expected columns."""
        expected_columns = {
            "game_id",
            "frame_id",
            "period_id",
            "timestamp",
            "ball_state",
            "ball_owning_team_id",
            "ball_x",
            "ball_y",
            "ball_z",
            "team_id",
            "player_id",
            "x",
            "y",
            "z",
        }
        assert set(dataset.tracking.columns) == expected_columns

    def test_no_ball_rows(self, dataset):
        """Test that long_ball format has no ball rows."""
        ball_count = dataset.tracking.filter("team_id = 'ball'").count()
        assert ball_count == 0


class TestPySparkTrackingWide:
    """Tests for tracking DataFrame with 'wide' layout and PySpark engine."""

    @pytest.fixture
    def dataset(self, spark):
        """Load dataset with PySpark engine and wide layout."""
        return secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            layout="wide",
            engine="pyspark",
            spark_session=spark,
        )

    def test_base_columns(self, dataset):
        """Test that wide format has expected base columns."""
        base_columns = {
            "frame_id",
            "period_id",
            "timestamp",
            "ball_state",
            "ball_owning_team_id",
            "ball_x",
            "ball_y",
            "ball_z",
        }
        assert base_columns.issubset(set(dataset.tracking.columns))

    def test_player_columns(self, dataset):
        """Test that wide format has player coordinate columns."""
        columns = dataset.tracking.columns
        x_columns = [c for c in columns if c.endswith("_x") and c != "ball_x"]
        y_columns = [c for c in columns if c.endswith("_y") and c != "ball_y"]
        z_columns = [c for c in columns if c.endswith("_z") and c != "ball_z"]

        assert len(x_columns) == 23
        assert len(x_columns) == len(y_columns) == len(z_columns)

    def test_one_row_per_frame(self, dataset):
        """Test that wide format has exactly one row per frame."""
        total_rows = dataset.tracking.count()
        unique_frames = dataset.tracking.select("frame_id").distinct().count()
        assert total_rows == unique_frames


class TestPySparkOnlyAlive:
    """Tests for only_alive parameter with PySpark engine."""

    def test_only_alive_filters_dead_frames(self, spark):
        """Test that only_alive=True filters out dead ball frames."""
        dataset_all = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            only_alive=False,
            exclude_missing_ball_frames=False,
            engine="pyspark",
            spark_session=spark,
        )
        dataset_alive = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            only_alive=True,
            exclude_missing_ball_frames=False,
            engine="pyspark",
            spark_session=spark,
        )

        # Same values as Polars tests
        assert dataset_all.tracking.count() == 4600
        assert dataset_alive.tracking.count() == 4554

    def test_only_alive_no_dead_frames(self, spark):
        """Test that only_alive=True results in no dead ball frames."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            only_alive=True,
            engine="pyspark",
            spark_session=spark,
        )

        dead_count = dataset.tracking.filter("ball_state = 'dead'").count()
        assert dead_count == 0


class TestPySparkConversion:
    """Tests for conversion between Polars and PySpark."""

    def test_to_polars_conversion(self, spark):
        """Test conversion from PySpark back to Polars."""
        import polars as pl

        dataset_spark = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="pyspark",
            spark_session=spark,
        )

        dataset_polars = dataset_spark.to_polars()

        assert dataset_polars.engine == "polars"
        assert isinstance(dataset_polars.tracking, pl.DataFrame)
        assert isinstance(dataset_polars.metadata, pl.DataFrame)
        assert isinstance(dataset_polars.teams, pl.DataFrame)
        assert isinstance(dataset_polars.players, pl.DataFrame)
        assert isinstance(dataset_polars.periods, pl.DataFrame)

    def test_to_pyspark_conversion(self, spark):
        """Test conversion from Polars to PySpark."""
        dataset_polars = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="polars",
        )

        dataset_spark = dataset_polars.to_pyspark(spark)

        assert dataset_spark.engine == "pyspark"
        assert isinstance(dataset_spark.tracking, SparkDataFrame)
        assert isinstance(dataset_spark.metadata, SparkDataFrame)

    def test_to_polars_idempotent(self, spark):
        """Test that to_polars() on Polars dataset returns self."""
        import polars as pl

        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="polars",
        )

        result = dataset.to_polars()
        assert result is dataset

    def test_to_pyspark_idempotent(self, spark):
        """Test that to_pyspark() on PySpark dataset returns self."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="pyspark",
            spark_session=spark,
        )

        result = dataset.to_pyspark(spark)
        assert result is dataset

    def test_round_trip_preserves_data(self, spark):
        """Test that Polars -> PySpark -> Polars preserves data."""
        import polars as pl

        # Load with Polars
        dataset_polars1 = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="polars",
        )

        # Convert to PySpark and back
        dataset_spark = dataset_polars1.to_pyspark(spark)
        dataset_polars2 = dataset_spark.to_polars()

        # Row counts should match
        assert len(dataset_polars2.tracking) == len(dataset_polars1.tracking)
        assert len(dataset_polars2.metadata) == len(dataset_polars1.metadata)
        assert len(dataset_polars2.teams) == len(dataset_polars1.teams)
        assert len(dataset_polars2.players) == len(dataset_polars1.players)
        assert len(dataset_polars2.periods) == len(dataset_polars1.periods)


class TestPySparkRepr:
    """Tests for TrackingDataset repr with PySpark engine."""

    def test_repr_includes_engine(self, spark):
        """Test that repr includes engine information."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            engine="pyspark",
            spark_session=spark,
        )

        repr_str = repr(dataset)
        assert "pyspark" in repr_str
        assert "TrackingDataset" in repr_str


class TestPySparkSchemaComparison:
    """Tests verifying schema compatibility between Polars and PySpark."""

    @pytest.fixture
    def datasets(self, spark):
        """Load same data with both engines."""
        polars_ds = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, engine="polars"
        )
        pyspark_ds = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, engine="pyspark", spark_session=spark
        )
        return polars_ds, pyspark_ds

    def test_tracking_columns_match(self, datasets):
        """Test that tracking DataFrames have identical columns."""
        polars_ds, pyspark_ds = datasets
        assert set(polars_ds.tracking.columns) == set(pyspark_ds.tracking.columns)

    def test_metadata_columns_match(self, datasets):
        """Test that metadata DataFrames have identical columns."""
        polars_ds, pyspark_ds = datasets
        assert set(polars_ds.metadata.columns) == set(pyspark_ds.metadata.columns)

    def test_teams_columns_match(self, datasets):
        """Test that teams DataFrames have identical columns."""
        polars_ds, pyspark_ds = datasets
        assert set(polars_ds.teams.columns) == set(pyspark_ds.teams.columns)

    def test_players_columns_match(self, datasets):
        """Test that players DataFrames have identical columns."""
        polars_ds, pyspark_ds = datasets
        assert set(polars_ds.players.columns) == set(pyspark_ds.players.columns)

    def test_periods_columns_match(self, datasets):
        """Test that periods DataFrames have identical columns."""
        polars_ds, pyspark_ds = datasets
        assert set(polars_ds.periods.columns) == set(pyspark_ds.periods.columns)

    def test_uint_to_signed_int_mapping(self, datasets):
        """Test that unsigned ints are correctly mapped to signed for Arrow optimization.

        PySpark's Arrow path doesn't support unsigned integers, so we cast:
        - UInt8 -> Int16 (SmallIntType)
        - UInt16 -> Int32 (IntegerType)
        - UInt32 -> Int64 (LongType)
        - UInt64 -> Int64 (LongType)

        This enables 11-34x speedup by allowing Arrow optimization.
        """
        from pyspark.sql.types import LongType, IntegerType, ShortType
        import polars as pl

        polars_ds, pyspark_ds = datasets

        # Build expected type mapping
        polars_to_spark_type = {
            pl.UInt8: ShortType,      # Int16
            pl.UInt16: IntegerType,   # Int32
            pl.UInt32: LongType,      # Int64
            pl.UInt64: LongType,      # Int64
        }

        # Check tracking DataFrame for any unsigned int columns
        polars_schema = dict(zip(polars_ds.tracking.columns, polars_ds.tracking.dtypes))
        spark_schema = {f.name: type(f.dataType) for f in pyspark_ds.tracking.schema.fields}

        for col_name, polars_dtype in polars_schema.items():
            if polars_dtype in polars_to_spark_type:
                expected_spark_type = polars_to_spark_type[polars_dtype]
                actual_spark_type = spark_schema[col_name]
                assert actual_spark_type == expected_spark_type, (
                    f"Column {col_name}: expected {expected_spark_type.__name__}, "
                    f"got {actual_spark_type.__name__}"
                )


class TestPySparkErrorHandling:
    """Tests for error handling with PySpark engine."""

    def test_invalid_engine_raises_error(self):
        """Test that invalid engine raises ValueError."""
        with pytest.raises(ValueError, match="Invalid engine"):
            secondspectrum.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH,
                engine="invalid",
            )
