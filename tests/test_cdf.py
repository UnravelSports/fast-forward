"""Integration tests for CDF (Common Data Format) tracking data loading."""

import pytest
import polars as pl

from fastforward import cdf
from tests.config import (
    CDF_RAW as RAW_DATA_PATH,
    CDF_META as META_DATA_PATH,
)


class TestLoadTracking:
    """Tests for cdf.load_tracking function."""

    def test_returns_tracking_dataset(self):
        """Test that load_tracking returns a TrackingDataset object."""
        from fastforward._dataset import TrackingDataset

        result = cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

        assert isinstance(result, TrackingDataset)
        assert isinstance(result.tracking, pl.DataFrame)
        assert isinstance(result.metadata, pl.DataFrame)
        assert isinstance(result.teams, pl.DataFrame)
        assert isinstance(result.players, pl.DataFrame)
        assert isinstance(result.periods, pl.DataFrame)


class TestMetadataDataFrame:
    """Tests for the metadata DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def metadata_df(self, dataset):
        """Load and return the metadata DataFrame."""
        return dataset.metadata

    def test_single_row(self, metadata_df):
        """Test that metadata_df contains exactly one row."""
        assert metadata_df.height == 1

    def test_schema(self, metadata_df):
        """Test that metadata_df has expected columns."""
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
        assert set(metadata_df.columns) == expected_columns

    def test_provider_value(self, metadata_df):
        """Test provider is 'cdf'."""
        assert metadata_df["provider"][0] == "cdf"

    def test_coordinate_system_value(self, metadata_df):
        """Test coordinate_system is 'cdf'."""
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_orientation_value(self, metadata_df):
        """Test orientation is 'static_home_away'."""
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_team_names(self, metadata_df):
        """Test team names are extracted correctly."""
        assert metadata_df["home_team"][0] == "Brisbane Roar FC"
        assert metadata_df["away_team"][0] == "Perth Glory Football Club"

    def test_team_ids(self, metadata_df):
        """Test team IDs are extracted correctly."""
        assert metadata_df["home_team_id"][0] == "1802"
        assert metadata_df["away_team_id"][0] == "871"

    def test_pitch_dimensions(self, metadata_df):
        """Test pitch dimensions are correct."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert pitch_length == pytest.approx(105.0, rel=0.01)
        assert pitch_width == pytest.approx(68.0, rel=0.01)

    def test_fps(self, metadata_df):
        """Test fps is 10."""
        assert metadata_df["fps"][0] == pytest.approx(10.0, rel=0.01)

    def test_game_id(self, metadata_df):
        """Test game_id is present."""
        assert metadata_df["game_id"][0] == "1925299"


class TestTeamDataFrame:
    """Tests for the team DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def team_df(self, dataset):
        """Load and return the team DataFrame."""
        return dataset.teams

    def test_two_rows(self, team_df):
        """Test that team_df contains exactly two rows."""
        assert team_df.height == 2

    def test_schema(self, team_df):
        """Test that team_df has expected columns."""
        expected_columns = {"game_id", "team_id", "name", "ground"}
        assert set(team_df.columns) == expected_columns

    def test_grounds(self, team_df):
        """Test that team_df contains home and away teams."""
        grounds = set(team_df["ground"].to_list())
        assert grounds == {"home", "away"}

    def test_team_names(self, team_df):
        """Test team names are correct."""
        home_team = team_df.filter(pl.col("ground") == "home")
        away_team = team_df.filter(pl.col("ground") == "away")
        assert home_team["name"][0] == "Brisbane Roar FC"
        assert away_team["name"][0] == "Perth Glory Football Club"


class TestPlayerDataFrame:
    """Tests for the player DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def player_df(self, dataset):
        """Load and return the player DataFrame."""
        return dataset.players

    def test_schema(self, player_df):
        """Test that player_df has expected columns."""
        expected_columns = {
            "game_id",
            "team_id",
            "player_id",
            "name",
            "first_name",
            "last_name",
            "jersey_number",
            "position",
            "is_starter",
        }
        assert set(player_df.columns) == expected_columns

    def test_has_players(self, player_df):
        """Test that player_df contains players from both teams."""
        # CDF test data has 18 home + 18 away = 36 players
        assert player_df.height == 36

    def test_home_players(self, player_df):
        """Test home team players."""
        home_players = player_df.filter(pl.col("team_id") == "1802")
        assert home_players.height == 18

    def test_away_players(self, player_df):
        """Test away team players."""
        away_players = player_df.filter(pl.col("team_id") == "871")
        assert away_players.height == 18

    def test_position_standardized(self, player_df):
        """Test that positions are standardized codes."""
        valid_positions = {
            "GK",
            "LB",
            "RB",
            "LCB",
            "CB",
            "RCB",
            "LWB",
            "RWB",
            "LDM",
            "CDM",
            "RDM",
            "LCM",
            "CM",
            "RCM",
            "LAM",
            "CAM",
            "RAM",
            "LW",
            "RW",
            "LM",
            "RM",
            "LF",
            "ST",
            "RF",
            "CF",
            "SUB",
            "UNK",
        }
        positions = set(player_df["position"].to_list())
        assert positions.issubset(valid_positions)

    def test_starter_flag(self, player_df):
        """Test is_starter flag."""
        starters = player_df.filter(pl.col("is_starter") == True)
        # Expect 11 starters per team = 22
        assert starters.height == 22

    def test_jersey_numbers(self, player_df):
        """Test jersey numbers are set."""
        # All jersey numbers should be > 0
        assert player_df.filter(pl.col("jersey_number") > 0).height > 0


class TestTrackingDataFrameLong:
    """Tests for tracking DataFrame with 'long' layout."""

    @pytest.fixture
    def dataset(self):
        """Load tracking data with long layout."""
        return cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, layout="long", lazy=False)

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking DataFrame."""
        return dataset.tracking

    def test_schema(self, tracking_df):
        """Test that long format has expected columns."""
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
        assert set(tracking_df.columns) == expected_columns

    def test_has_ball_rows(self, tracking_df):
        """Test that long format includes ball as separate rows."""
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        # 100 frames total (50 first half + 51 second half)
        # Some frames may be filtered by ball_state
        assert ball_rows.height > 0

        # Check ball rows have player_id = "ball"
        assert all(pid == "ball" for pid in ball_rows["player_id"].to_list())

    def test_timestamp_type(self, tracking_df):
        """Test that timestamp is Duration type."""
        assert tracking_df.schema["timestamp"] == pl.Duration("ms")

    def test_has_multiple_periods(self, tracking_df):
        """Test that data includes two periods."""
        periods = tracking_df["period_id"].unique().to_list()
        assert len(periods) == 2


class TestTrackingDataFrameLongBall:
    """Tests for tracking DataFrame with 'long_ball' layout."""

    @pytest.fixture
    def dataset(self):
        """Load tracking data with long_ball layout."""
        return cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="long_ball", lazy=False
        )

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking DataFrame."""
        return dataset.tracking

    def test_schema(self, tracking_df):
        """Test that long_ball format has expected columns."""
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
        assert set(tracking_df.columns) == expected_columns

    def test_no_ball_rows(self, tracking_df):
        """Test that long_ball format has no ball rows."""
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 0


class TestTrackingDataFrameWide:
    """Tests for tracking DataFrame with 'wide' layout."""

    @pytest.fixture
    def dataset(self):
        """Load tracking data with wide layout."""
        return cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, layout="wide", lazy=False)

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking DataFrame."""
        return dataset.tracking

    def test_base_columns(self, tracking_df):
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
        assert base_columns.issubset(set(tracking_df.columns))

    def test_player_columns(self, tracking_df):
        """Test that wide format has player coordinate columns."""
        columns = tracking_df.columns
        # Should have columns like "{player_id}_x", "{player_id}_y", "{player_id}_z"
        x_columns = [c for c in columns if c.endswith("_x") and c != "ball_x"]
        y_columns = [c for c in columns if c.endswith("_y") and c != "ball_y"]
        z_columns = [c for c in columns if c.endswith("_z") and c != "ball_z"]

        assert len(x_columns) > 0
        assert len(x_columns) == len(y_columns) == len(z_columns)

    def test_one_row_per_frame(self, tracking_df):
        """Test that wide format has exactly one row per frame."""
        frame_count = tracking_df["frame_id"].n_unique()
        assert tracking_df.height == frame_count


class TestCoordinateSystem:
    """Tests for coordinate system parameter."""

    def test_cdf_coordinates(self):
        """Test that CDF coordinates work correctly."""
        dataset = cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="cdf", lazy=False
        )
        assert dataset.tracking.height > 0
        assert dataset.metadata["coordinate_system"][0] == "cdf"

    def test_invalid_coordinate_system(self):
        """Test that invalid coordinate system raises error."""
        with pytest.raises(Exception):
            cdf.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH, coordinates="invalid", lazy=False
            )


class TestLayoutParameter:
    """Tests for layout parameter validation."""

    def test_invalid_layout(self):
        """Test that invalid layout raises error."""
        with pytest.raises(Exception):
            cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, layout="invalid", lazy=False)


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_tracking_file(self):
        """Test that missing tracking file raises error."""
        with pytest.raises(Exception):
            cdf.load_tracking("nonexistent_tracking.jsonl", META_DATA_PATH, lazy=False)

    def test_missing_metadata_file(self):
        """Test that missing metadata file raises error."""
        with pytest.raises(Exception):
            cdf.load_tracking(RAW_DATA_PATH, "nonexistent_metadata.json", lazy=False)


class TestOnlyAliveParameter:
    """Tests for only_alive parameter."""

    def test_only_alive_filters_dead_frames(self):
        """Test that only_alive=True filters out dead ball frames."""
        dataset_all = cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=False, lazy=False
        )
        dataset_alive = cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True, lazy=False
        )

        # All frames have ball_status=false in our test data, so alive should be empty
        # or equal if no dead frames
        assert dataset_all.tracking.height >= dataset_alive.tracking.height

    def test_only_alive_no_dead_frames(self):
        """Test that only_alive=True results in no dead ball frames."""
        dataset = cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True, lazy=False
        )

        # All rows should be alive
        dead_rows = dataset.tracking.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0


class TestExcludeMissingBallFrames:
    """Tests for exclude_missing_ball_frames parameter."""

    def test_exclude_missing_ball_frames_default_true(self):
        """Default exclude_missing_ball_frames=True excludes frame 30281."""
        dataset = cdf.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            only_alive=False,
            lazy=False,
        )
        frame_ids = dataset.tracking["frame_id"].unique().to_list()
        assert 30281 not in frame_ids

    def test_exclude_missing_ball_frames_false_includes_all(self):
        """exclude_missing_ball_frames=False includes frame 30281."""
        dataset = cdf.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            only_alive=False,
            exclude_missing_ball_frames=False,
            lazy=False,
        )
        frame_ids = dataset.tracking["frame_id"].unique().to_list()
        assert 30281 in frame_ids

    def test_missing_ball_frame_has_nan_coordinates(self):
        """Missing ball frames have NaN coordinates when not excluded."""
        dataset = cdf.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            only_alive=False,
            exclude_missing_ball_frames=False,
            layout="long",
            lazy=False,
        )
        ball_frame = dataset.tracking.filter(
            (pl.col("team_id") == "ball") & (pl.col("frame_id") == 30281)
        )
        assert len(ball_frame) == 1
        assert ball_frame["x"].is_nan()[0]
        assert ball_frame["y"].is_nan()[0]
        assert ball_frame["z"].is_nan()[0]

    def test_exclude_missing_ball_frames_filters_correct_count(self):
        """Filtering removes 1 frame worth of rows."""
        dataset_all = cdf.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            only_alive=False,
            exclude_missing_ball_frames=False,
            lazy=False,
        )
        dataset_filtered = cdf.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            only_alive=False,
            exclude_missing_ball_frames=True,
            lazy=False,
        )
        # 102 frames vs 101 frames - each frame has 23 entities (11+11+1 ball)
        assert dataset_all.tracking.height > dataset_filtered.tracking.height


class TestOrientationParameter:
    """Tests for orientation parameter."""

    def test_orientation_default_static_home_away(self):
        """Test that orientation defaults to 'static_home_away'."""
        dataset = cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_orientation_attack_left(self):
        """Test orientation 'attack_left'."""
        dataset = cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_left", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "attack_left"


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyParameter:
    """Tests for lazy loading parameter."""

    def test_lazy_false_returns_dataframe(self):
        """Test that lazy=False returns a DataFrame."""
        dataset = cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert isinstance(dataset.tracking, pl.DataFrame)

    def test_lazy_true_returns_lazyframe(self):
        """Test that lazy=True returns a LazyFrame."""
        dataset = cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        assert isinstance(dataset.tracking, pl.LazyFrame)

    def test_lazy_collect_works(self):
        """Test that collecting a LazyFrame works."""
        dataset = cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        collected = dataset.tracking.collect()
        assert isinstance(collected, pl.DataFrame)
        assert collected.height > 0


class TestLazyNotImplemented:
    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

    def test_from_cache_raises(self):
        with pytest.raises(NotImplementedError, match="cache loading"):
            cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, from_cache=True)


class TestIncludeGameId:
    """Tests for include_game_id parameter."""

    def test_include_game_id_true(self):
        """Test that include_game_id=True adds game_id column."""
        dataset = cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, include_game_id=True, lazy=False
        )
        assert "game_id" in dataset.tracking.columns
        assert dataset.tracking["game_id"][0] == "1925299"

    def test_include_game_id_false(self):
        """Test that include_game_id=False removes game_id column."""
        dataset = cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, include_game_id=False, lazy=False
        )
        assert "game_id" not in dataset.tracking.columns

    def test_include_game_id_custom_string(self):
        """Test that include_game_id=str uses custom value."""
        custom_id = "custom_game_123"
        dataset = cdf.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, include_game_id=custom_id, lazy=False
        )
        assert "game_id" in dataset.tracking.columns
        assert dataset.tracking["game_id"][0] == custom_id


class TestCDFCoordinates:
    """Tests specific to CDF coordinate handling."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    def test_coordinates_center_origin(self, dataset):
        """Test that CDF coordinates have center origin (values can be negative)."""
        tracking_df = dataset.tracking
        # Filter to player data (not ball)
        player_data = tracking_df.filter(pl.col("team_id") != "ball")

        # CDF coordinates should have both positive and negative values
        x_values = player_data["x"].to_list()
        y_values = player_data["y"].to_list()

        # Check we have both positive and negative x values
        has_positive_x = any(x > 0 for x in x_values)
        has_negative_x = any(x < 0 for x in x_values)
        assert has_positive_x and has_negative_x

        # Check we have both positive and negative y values
        has_positive_y = any(y > 0 for y in y_values)
        has_negative_y = any(y < 0 for y in y_values)
        assert has_positive_y and has_negative_y

    def test_ball_coordinates(self):
        """Test ball coordinates are parsed correctly."""
        # Load with only_alive=False to get all frames including frame 0
        dataset = cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False, only_alive=False)
        tracking_df = dataset.tracking
        ball_data = tracking_df.filter(pl.col("team_id") == "ball")

        # First frame ball position (frame_id=0)
        frame_0_ball = ball_data.filter(pl.col("frame_id") == 0).row(0, named=True)
        assert frame_0_ball["x"] == pytest.approx(0.66, abs=0.01)
        assert frame_0_ball["y"] == pytest.approx(-0.58, abs=0.01)
        assert frame_0_ball["z"] == pytest.approx(0.29, abs=0.01)


class TestPeriodsDataFrame:
    """Tests for the periods DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return cdf.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def periods_df(self, dataset):
        """Load and return the periods DataFrame."""
        return dataset.periods

    def test_has_two_periods(self, periods_df):
        """Test that periods_df has 2 periods."""
        assert periods_df.height == 2

    def test_schema(self, periods_df):
        """Test that periods_df has expected columns."""
        expected_columns = {
            "game_id",
            "period_id",
            "start_frame_id",
            "end_frame_id",
            "start_timestamp",
            "end_timestamp",
            "duration",
        }
        assert set(periods_df.columns) == expected_columns

    def test_period_ids(self, periods_df):
        """Test period IDs are 1 and 2."""
        period_ids = sorted(periods_df["period_id"].to_list())
        assert period_ids == [1, 2]

    def test_period_timing(self, periods_df):
        """Test that all periods have correct timing values."""
        from datetime import timedelta

        periods = periods_df.sort("period_id")

        # Period 1
        p1 = periods.row(0, named=True)
        assert p1["start_timestamp"] == timedelta(milliseconds=2000)
        assert p1["end_timestamp"] == timedelta(milliseconds=3900)
        assert p1["duration"] == timedelta(milliseconds=1900)

        # Period 2
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] == timedelta(milliseconds=-3022700)
        assert p2["end_timestamp"] == timedelta(milliseconds=-3020000)
        assert p2["duration"] == timedelta(milliseconds=2700)
