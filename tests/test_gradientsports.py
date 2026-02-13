"""Integration tests for GradientSports (PFF) tracking data loading."""

import pytest
import polars as pl

from fastforward import gradientsports
from tests.config import (
    GS_RAW as RAW_DATA_PATH,
    GS_META as META_DATA_PATH,
    GS_ROSTER as ROSTER_DATA_PATH,
)


class TestLoadTracking:
    """Tests for gradientsports.load_tracking function."""

    def test_returns_tracking_dataset(self):
        """Test that load_tracking returns a TrackingDataset object."""
        from fastforward._dataset import TrackingDataset

        result = gradientsports.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=False
        )

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
        return gradientsports.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=False
        )

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
        """Test provider is 'gradientsports'."""
        assert metadata_df["provider"][0] == "gradientsports"

    def test_coordinate_system_value(self, metadata_df):
        """Test coordinate_system is 'gradientsports'."""
        assert metadata_df["coordinate_system"][0] == "gradientsports"

    def test_orientation_value(self, metadata_df):
        """Test orientation is 'static_home_away'."""
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_team_names(self, metadata_df):
        """Test team names are extracted correctly."""
        assert metadata_df["home_team"][0] == "Argentina"
        assert metadata_df["away_team"][0] == "France"

    def test_team_ids(self, metadata_df):
        """Test team IDs are extracted correctly."""
        assert metadata_df["home_team_id"][0] == "364"
        assert metadata_df["away_team_id"][0] == "363"

    def test_pitch_dimensions(self, metadata_df):
        """Test pitch dimensions are correct (105m x 68m)."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert pitch_length == pytest.approx(105.0, rel=0.01)
        assert pitch_width == pytest.approx(68.0, rel=0.01)

    def test_fps(self, metadata_df):
        """Test fps is ~29.97."""
        assert metadata_df["fps"][0] == pytest.approx(29.97, rel=0.01)

    def test_game_id(self, metadata_df):
        """Test game_id is present."""
        assert metadata_df["game_id"][0] == "10517"

    def test_game_date(self, metadata_df):
        """Test game date (2022-12-18 World Cup Final)."""
        from datetime import date

        assert metadata_df["game_date"][0] == date(2022, 12, 18)


class TestTeamDataFrame:
    """Tests for the team DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return gradientsports.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=False
        )

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
        assert home_team["name"][0] == "Argentina"
        assert away_team["name"][0] == "France"


class TestPlayerDataFrame:
    """Tests for the player DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return gradientsports.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=False
        )

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
        """Test that player_df contains 50 players (25 per team)."""
        assert player_df.height == 50

    def test_home_players(self, player_df):
        """Test home team (Argentina) players."""
        home_players = player_df.filter(pl.col("team_id") == "364")
        assert home_players.height == 26

    def test_away_players(self, player_df):
        """Test away team (France) players."""
        away_players = player_df.filter(pl.col("team_id") == "363")
        assert away_players.height == 24

    def test_specific_home_player(self, player_df):
        """Test specific home player (Nahuel Molina)."""
        molina = player_df.filter(pl.col("player_id") == "13222")
        assert molina.height == 1
        assert molina["name"][0] == "Nahuel Molina"
        assert molina["team_id"][0] == "364"
        assert molina["jersey_number"][0] == 26
        assert molina["position"][0] == "RB"

    def test_specific_away_player(self, player_df):
        """Test specific away player (Kingsley Coman)."""
        coman = player_df.filter(pl.col("player_id") == "4622")
        assert coman.height == 1
        assert coman["name"][0] == "Kingsley Coman"
        assert coman["team_id"][0] == "363"
        assert coman["jersey_number"][0] == 20

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
        """Test is_starter flag - 11 starters per team = 22 total."""
        starters = player_df.filter(pl.col("is_starter") == True)
        assert starters.height == 22


class TestTrackingDataFrameLong:
    """Tests for tracking DataFrame with 'long' layout."""

    @pytest.fixture
    def dataset(self):
        """Load tracking data with long layout (complete frames only)."""
        return gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            layout="long",
            only_alive=False,
            include_incomplete_frames=False,
            lazy=False,
        )

    @pytest.fixture
    def dataset_all_frames(self):
        """Load tracking data with all frames including incomplete."""
        return gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            layout="long",
            only_alive=False,
            include_incomplete_frames=True,
            lazy=False,
        )

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

    def test_has_ball_rows(self, dataset_all_frames):
        """Test that long format includes ball as separate rows."""
        tracking_df = dataset_all_frames.tracking
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 200

        # Check ball rows have player_id = "ball"
        assert all(pid == "ball" for pid in ball_rows["player_id"].to_list())

    def test_timestamp_type(self, tracking_df):
        """Test that timestamp is Duration type."""
        assert tracking_df.schema["timestamp"] == pl.Duration("ms")

    def test_has_four_periods(self, dataset_all_frames):
        """Test that data includes 4 periods (World Cup Final - extra time)."""
        tracking_df = dataset_all_frames.tracking
        periods = tracking_df["period_id"].unique().to_list()
        assert len(periods) == 4

    def test_frame_count(self, dataset_all_frames):
        """Test total number of frames.

        The test data has 200 unique frames across 4 periods when including incomplete frames.
        """
        tracking_df = dataset_all_frames.tracking
        unique_frames = tracking_df["frame_id"].n_unique()
        assert unique_frames == 200


class TestTrackingDataFrameLongBall:
    """Tests for tracking DataFrame with 'long_ball' layout."""

    @pytest.fixture
    def dataset(self):
        """Load tracking data with long_ball layout."""
        return gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            layout="long_ball",
            only_alive=False,
            lazy=False,
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
        """Load tracking data with wide layout (all frames for column count)."""
        return gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            layout="wide",
            only_alive=False,
            include_incomplete_frames=True,
            lazy=False,
        )

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

        assert len(x_columns) == 35
        assert len(x_columns) == len(y_columns) == len(z_columns)

    def test_one_row_per_frame(self, tracking_df):
        """Test that wide format has exactly one row per frame."""
        frame_count = tracking_df["frame_id"].n_unique()
        assert tracking_df.height == frame_count


class TestCoordinateSystem:
    """Tests for coordinate system parameter."""

    def test_gradientsports_coordinates(self):
        """Test that gradientsports coordinates work correctly."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            coordinates="gradientsports",
            lazy=False,
        )
        # Default include_incomplete_frames=False filters incomplete frames
        assert dataset.tracking.height == 3380
        assert dataset.metadata["coordinate_system"][0] == "gradientsports"

    def test_pff_coordinates_alias(self):
        """Test that 'pff' is an alias for gradientsports coordinates."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            coordinates="pff",
            lazy=False,
        )
        assert dataset.tracking.height == 3380
        assert dataset.metadata["coordinate_system"][0] == "pff"

    def test_cdf_coordinates(self):
        """Test CDF coordinate system."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            coordinates="cdf",
            lazy=False,
        )
        assert dataset.tracking.height == 3380
        assert dataset.metadata["coordinate_system"][0] == "cdf"


class TestLayoutParameter:
    """Tests for layout parameter validation."""

    def test_invalid_layout(self):
        """Test that invalid layout raises error."""
        with pytest.raises(Exception):
            gradientsports.load_tracking(
                RAW_DATA_PATH,
                META_DATA_PATH,
                ROSTER_DATA_PATH,
                layout="invalid",
                lazy=False,
            )


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_tracking_file(self):
        """Test that missing tracking file raises error."""
        with pytest.raises(Exception):
            gradientsports.load_tracking(
                "nonexistent_tracking.jsonl",
                META_DATA_PATH,
                ROSTER_DATA_PATH,
                lazy=False,
            )

    def test_missing_metadata_file(self):
        """Test that missing metadata file raises error."""
        with pytest.raises(Exception):
            gradientsports.load_tracking(
                RAW_DATA_PATH,
                "nonexistent_metadata.json",
                ROSTER_DATA_PATH,
                lazy=False,
            )

    def test_missing_roster_file(self):
        """Test that missing roster file raises error."""
        with pytest.raises(Exception):
            gradientsports.load_tracking(
                RAW_DATA_PATH,
                META_DATA_PATH,
                "nonexistent_roster.json",
                lazy=False,
            )


class TestOnlyAliveParameter:
    """Tests for only_alive parameter."""

    def test_only_alive_filters_dead_frames(self):
        """Test that only_alive=True filters out dead ball frames."""
        # Using include_incomplete_frames=True to get all frames for comparison
        dataset_all = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            only_alive=False,
            include_incomplete_frames=True,
            lazy=False,
        )
        dataset_alive = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            only_alive=True,
            include_incomplete_frames=True,
            lazy=False,
        )

        # Exact row counts for test data (with include_incomplete_frames=True)
        assert dataset_all.tracking.height == 4577
        assert dataset_alive.tracking.height == 4531

    def test_only_alive_no_dead_frames(self):
        """Test that only_alive=True results in no dead ball frames."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            only_alive=True,
            lazy=False,
        )

        # All rows should be alive
        dead_rows = dataset.tracking.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0


class TestOrientationParameter:
    """Tests for orientation parameter."""

    def test_orientation_default_static_home_away(self):
        """Test that orientation defaults to 'static_home_away'."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=False
        )
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_orientation_attack_left(self):
        """Test orientation 'attack_left'."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            orientation="attack_left",
            lazy=False,
        )
        assert dataset.metadata["orientation"][0] == "attack_left"


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyParameter:
    """Tests for lazy loading parameter."""

    def test_lazy_false_returns_dataframe(self):
        """Test that lazy=False returns a DataFrame."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=False
        )
        assert isinstance(dataset.tracking, pl.DataFrame)

    def test_lazy_true_returns_lazyframe(self):
        """Test that lazy=True returns a LazyFrame."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=True
        )
        assert isinstance(dataset.tracking, pl.LazyFrame)

    def test_lazy_collect_works(self):
        """Test that collecting a LazyFrame works."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=True
        )
        collected = dataset.tracking.collect()
        assert isinstance(collected, pl.DataFrame)
        # Default include_incomplete_frames=False filters incomplete frames
        assert collected.height == 3380


class TestLazyNotImplemented:
    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            gradientsports.load_tracking(RAW_DATA_PATH, META_DATA_PATH, ROSTER_DATA_PATH, lazy=True)


class TestIncludeGameId:
    """Tests for include_game_id parameter."""

    def test_include_game_id_true(self):
        """Test that include_game_id=True adds game_id column."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_game_id=True,
            lazy=False,
        )
        assert "game_id" in dataset.tracking.columns
        assert dataset.tracking["game_id"][0] == "10517"

    def test_include_game_id_false(self):
        """Test that include_game_id=False removes game_id column."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_game_id=False,
            lazy=False,
        )
        assert "game_id" not in dataset.tracking.columns

    def test_include_game_id_custom_string(self):
        """Test that include_game_id=str uses custom value."""
        custom_id = "custom_game_123"
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_game_id=custom_id,
            lazy=False,
        )
        assert "game_id" in dataset.tracking.columns
        assert dataset.tracking["game_id"][0] == custom_id


class TestGradientSportsCoordinates:
    """Tests specific to GradientSports coordinate handling."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset with all frames for coordinate tests."""
        return gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            coordinates="gradientsports",
            only_alive=False,
            include_incomplete_frames=True,
            lazy=False,
        )

    def test_coordinates_center_origin(self, dataset):
        """Test that GradientSports coordinates have center origin."""
        tracking_df = dataset.tracking
        player_data = tracking_df.filter(pl.col("team_id") != "ball")

        x_values = player_data["x"].to_list()
        y_values = player_data["y"].to_list()

        # Check we have both positive and negative x values
        has_positive_x = any(x > 0 for x in x_values if x is not None)
        has_negative_x = any(x < 0 for x in x_values if x is not None)
        assert has_positive_x and has_negative_x

        # Check we have both positive and negative y values
        has_positive_y = any(y > 0 for y in y_values if y is not None)
        has_negative_y = any(y < 0 for y in y_values if y is not None)
        assert has_positive_y and has_negative_y

    def test_ball_coordinates(self, dataset):
        """Test ball coordinates are parsed correctly (first frame)."""
        tracking_df = dataset.tracking
        ball_data = tracking_df.filter(pl.col("team_id") == "ball")

        # First frame ball position (frame_id=4630)
        first_frame = ball_data.filter(pl.col("frame_id") == 4630).row(0, named=True)
        assert first_frame["x"] == pytest.approx(0.42, abs=0.01)
        assert first_frame["y"] == pytest.approx(1.59, abs=0.01)
        assert first_frame["z"] == pytest.approx(0.39, abs=0.01)

    def test_player_coordinates(self, dataset):
        """Test player coordinates match expected values."""
        tracking_df = dataset.tracking

        # Check home player Julian Alvarez (player_id=10715, jersey 9) in first frame
        first_frame = tracking_df.filter(
            (pl.col("frame_id") == 4630) & (pl.col("player_id") == "10715")
        )
        assert first_frame.height == 1
        row = first_frame.row(0, named=True)
        assert row["x"] == pytest.approx(4.987, abs=0.01)
        assert row["y"] == pytest.approx(-1.993, abs=0.01)


class TestPeriodsDataFrame:
    """Tests for the periods DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset with all frames for period boundary tests."""
        return gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_incomplete_frames=True,
            lazy=False,
        )

    @pytest.fixture
    def periods_df(self, dataset):
        """Load and return the periods DataFrame."""
        return dataset.periods

    def test_has_four_periods(self, periods_df):
        """Test that periods_df has 4 periods (World Cup Final with extra time)."""
        assert periods_df.height == 4

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
        """Test period IDs are 1, 2, 3, 4."""
        period_ids = sorted(periods_df["period_id"].to_list())
        assert period_ids == [1, 2, 3, 4]

    def test_first_period_frames(self, periods_df):
        """Test first period frame IDs."""
        period_1 = periods_df.filter(pl.col("period_id") == 1)
        assert period_1["start_frame_id"][0] == 4630
        assert period_1["end_frame_id"][0] == 99119

    def test_period_timing(self, periods_df):
        """Test that all periods have correct timing values."""
        from datetime import timedelta

        periods = periods_df.sort("period_id")

        # Period 1
        p1 = periods.row(0, named=True)
        assert p1["start_timestamp"] == timedelta(milliseconds=0)
        assert p1["end_timestamp"] == timedelta(milliseconds=3152786)
        assert p1["duration"] == timedelta(milliseconds=3152786)

        # Period 2
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] == timedelta(milliseconds=33)
        assert p2["end_timestamp"] == timedelta(milliseconds=3216549)
        assert p2["duration"] == timedelta(milliseconds=3216516)

        # Period 3
        p3 = periods.row(2, named=True)
        assert p3["start_timestamp"] == timedelta(milliseconds=0)
        assert p3["end_timestamp"] == timedelta(milliseconds=965165)
        assert p3["duration"] == timedelta(milliseconds=965165)

        # Period 4
        p4 = periods.row(3, named=True)
        assert p4["start_timestamp"] == timedelta(milliseconds=32)
        assert p4["end_timestamp"] == timedelta(milliseconds=1148614)
        assert p4["duration"] == timedelta(milliseconds=1148582)


class TestIncludeIncompleteFrames:
    """Tests for include_incomplete_frames parameter."""

    def test_include_incomplete_frames_false_default(self):
        """Test that include_incomplete_frames=False (default) filters incomplete frames."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_incomplete_frames=False,
            only_alive=False,
            lazy=False,
        )
        # Should filter out frames with null ball/player data
        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball")
        # ~97.5% of frames have null ball data, so we expect ~5 frames with complete ball data
        assert ball_rows.height == 148
        assert ball_rows.height < 200

    def test_include_incomplete_frames_true(self):
        """Test that include_incomplete_frames=True includes all frames."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_incomplete_frames=True,
            only_alive=False,
            lazy=False,
        )
        # Should include all 200 frames
        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 200

    def test_difference_in_frame_count(self):
        """Test that there's a meaningful difference between the two modes."""
        dataset_all = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_incomplete_frames=True,
            only_alive=False,
            lazy=False,
        )
        dataset_complete = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_incomplete_frames=False,
            only_alive=False,
            lazy=False,
        )
        # Significant difference expected
        all_frames = dataset_all.tracking["frame_id"].n_unique()
        complete_frames = dataset_complete.tracking["frame_id"].n_unique()
        assert all_frames == 200
        assert complete_frames == 148
        assert all_frames > complete_frames

    def test_incomplete_frames_have_default_ball(self):
        """Test that incomplete frames have ball at (0, 0, 0) when included."""
        dataset = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_incomplete_frames=True,
            only_alive=False,
            lazy=False,
        )
        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball")

        # Count frames where ball is at origin (0, 0, 0)
        origin_ball = ball_rows.filter(
            (pl.col("x") == 0.0) & (pl.col("y") == 0.0) & (pl.col("z") == 0.0)
        )
        # Most frames have null ball data which defaults to (0, 0, 0)
        assert origin_ball.height > 0

    def test_default_is_false(self):
        """Test that include_incomplete_frames defaults to False."""
        dataset_default = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            only_alive=False,
            lazy=False,
        )
        dataset_explicit = gradientsports.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            ROSTER_DATA_PATH,
            include_incomplete_frames=False,
            only_alive=False,
            lazy=False,
        )
        # Both should have the same number of rows
        assert dataset_default.tracking.height == dataset_explicit.tracking.height
        assert dataset_default.tracking.height == 3403
