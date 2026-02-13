"""Tests for Signality provider."""
import pytest
import polars as pl

from fastforward import signality
from tests.config import (
    SIG_META as META_DATA,
    SIG_VENUE as VENUE_INFO,
    SIG_RAW_FILES as RAW_DATA_FEEDS,
)


class TestSignalityBasic:
    """Basic Signality loading tests."""

    def test_load_with_list_of_files(self):
        """Test loading with list of file paths."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            only_alive=False,  # Include dead frames to get all 10 frames
        )

        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)

        # Based on the test data: 5 frames per period, 10 total frames
        # P1: 5 frames × (11 home + 11 away + 1 ball) = 115 rows
        # P2: 5 frames × (11 home + 10 away + 1 ball) = 110 rows
        # Total: 225 rows
        assert len(dataset.tracking) == 225
        assert len(dataset.metadata) == 1
        assert len(dataset.teams) == 2  # Home and away
        assert len(dataset.players) == 40  # 20 players per team

    def test_load_with_single_file(self):
        """Test loading with single file path."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS[0],  # Just period 1
            venue_information=VENUE_INFO,
            only_alive=False,
        )

        # P1 only: 5 frames × (11 home + 11 away + 1 ball) = 115 rows
        assert len(dataset.tracking) == 115
        assert len(dataset.metadata) == 1
        assert len(dataset.teams) == 2
        assert len(dataset.players) == 40


class TestSignalityColumns:
    """Test DataFrame column structure."""

    def test_tracking_columns(self):
        """Test tracking_df has expected columns."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            only_alive=False,
        )
        tracking_df = dataset.tracking

        expected_cols = [
            "game_id",
            "frame_id",
            "period_id",
            "timestamp",
            "ball_state",
            "team_id",
            "player_id",
            "x",
            "y",
            "z",
        ]
        assert all(col in tracking_df.columns for col in expected_cols)

    def test_metadata_columns(self):
        """Test metadata_df has expected columns."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
        )
        metadata_df = dataset.metadata

        expected_cols = [
            "provider",
            "game_id",
            "home_team",
            "home_team_id",
            "away_team",
            "away_team_id",
            "pitch_length",
            "pitch_width",
            "fps",
            "coordinate_system",
            "orientation",
        ]
        assert all(col in metadata_df.columns for col in expected_cols)

    def test_team_columns(self):
        """Test team_df has expected columns."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
        )
        team_df = dataset.teams

        expected_cols = ["game_id", "team_id", "name", "ground"]
        assert all(col in team_df.columns for col in expected_cols)

    def test_player_columns(self):
        """Test player_df has expected columns."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
        )
        player_df = dataset.players

        expected_cols = ["game_id", "team_id", "player_id", "jersey_number", "position"]
        assert all(col in player_df.columns for col in expected_cols)


class TestSignalityParameters:
    """Test various parameter options."""

    def test_only_alive_true(self):
        """Test only_alive=True filters dead ball frames."""
        dataset_all = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            only_alive=False,
        )
        tracking_df_all = dataset_all.tracking

        dataset_alive = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            only_alive=True,
        )
        tracking_df_alive = dataset_alive.tracking

        # alive: P1 4 frames × 23 rows + P2 4 frames × 22 rows = 92 + 88 = 180
        # all: 225 rows (includes dead frames)
        assert len(tracking_df_alive) == 180
        assert len(tracking_df_all) == 225
        # Verify all ball rows in alive df have ball_state == "alive"
        ball_rows = tracking_df_alive.filter(pl.col("team_id") == "ball")
        assert all(ball_rows["ball_state"] == "alive")

    def test_only_alive_false(self):
        """Test only_alive=False includes dead ball frames."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            only_alive=False,
        )
        tracking_df = dataset.tracking

        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        ball_states = ball_rows["ball_state"].unique().sort()
        # Should have both alive and dead
        assert "alive" in ball_states.to_list()
        assert "dead" in ball_states.to_list()

    def test_include_game_id_true(self):
        """Test include_game_id=True adds game_id column."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            include_game_id=True,
        )
        tracking_df = dataset.tracking
        team_df = dataset.teams
        player_df = dataset.players

        assert "game_id" in tracking_df.columns
        assert "game_id" in team_df.columns
        assert "game_id" in player_df.columns

    def test_include_game_id_false(self):
        """Test include_game_id=False omits game_id column."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            include_game_id=False,
        )
        tracking_df = dataset.tracking
        team_df = dataset.teams
        player_df = dataset.players

        assert "game_id" not in tracking_df.columns
        assert "game_id" not in team_df.columns
        assert "game_id" not in player_df.columns

    def test_include_game_id_custom(self):
        """Test include_game_id with custom string."""
        custom_id = "custom_game_123"
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            include_game_id=custom_id,
        )
        tracking_df = dataset.tracking

        assert "game_id" in tracking_df.columns
        assert tracking_df["game_id"][0] == custom_id

    def test_cdf_coordinates(self):
        """Test CDF coordinate system (default)."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            coordinates="cdf",
        )
        metadata_df = dataset.metadata

        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_signality_coordinates(self):
        """Test native Signality coordinates."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            coordinates="signality",
        )
        metadata_df = dataset.metadata

        # Signality coordinates are same as CDF
        assert metadata_df["coordinate_system"][0] == "signality"


class TestPeriodsDataFrame:
    """Tests for the periods DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            only_alive=False,
        )

    @pytest.fixture
    def periods_df(self, dataset):
        """Return the periods DataFrame."""
        return dataset.periods

    def test_has_two_periods(self, periods_df):
        """Test that periods_df has 2 periods."""
        assert periods_df.height == 2

    def test_schema(self, periods_df):
        """Test that periods_df has expected columns."""
        expected_columns = {
            "game_id", "period_id", "start_frame_id", "end_frame_id",
            "start_timestamp", "end_timestamp", "duration",
        }
        assert set(periods_df.columns) == expected_columns

    def test_period_timing(self, periods_df):
        """Test that all periods have correct timing values."""
        from datetime import timedelta

        periods = periods_df.sort("period_id")

        # Period 1
        p1 = periods.row(0, named=True)
        assert p1["start_timestamp"] == timedelta(milliseconds=12)
        assert p1["end_timestamp"] == timedelta(milliseconds=2922450)
        assert p1["duration"] == timedelta(milliseconds=2922438)

        # Period 2
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] == timedelta(milliseconds=12)
        assert p2["end_timestamp"] == timedelta(milliseconds=2965955)
        assert p2["duration"] == timedelta(milliseconds=2965943)


class TestSignalityData:
    """Test actual data content."""

    def test_pitch_dimensions(self):
        """Test pitch dimensions from venue info."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
        )
        metadata_df = dataset.metadata

        assert metadata_df["pitch_length"][0] == pytest.approx(104.82)
        assert metadata_df["pitch_width"][0] == pytest.approx(68.407)

    def test_team_info(self):
        """Test team information."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
        )
        team_df = dataset.teams

        assert len(team_df) == 2
        # Check home team
        home_team = team_df.filter(pl.col("ground") == "home")
        assert len(home_team) == 1
        assert home_team["name"][0] == "Home Team"

        # Check away team
        away_team = team_df.filter(pl.col("ground") == "away")
        assert len(away_team) == 1
        assert away_team["name"][0] == "Away Team"

    def test_player_count(self):
        """Test player counts per team."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
        )
        player_df = dataset.players

        # 20 players per team = 40 total
        assert len(player_df) == 40

        home_players = player_df.filter(pl.col("team_id") == "home")
        away_players = player_df.filter(pl.col("team_id") == "away")
        assert len(home_players) == 20
        assert len(away_players) == 20

    def test_frame_data_values(self):
        """Test specific frame data values match expected."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            coordinates="signality",
            only_alive=False,
        )
        tracking_df = dataset.tracking

        # Get first frame (frame_id=0)
        first_frame = tracking_df.filter(pl.col("frame_id") == 0)

        # Ball position at first frame
        ball_row = first_frame.filter(pl.col("team_id") == "ball")
        assert len(ball_row) == 1
        assert ball_row["x"][0] == pytest.approx(-41.18)
        assert ball_row["y"][0] == pytest.approx(0.08)
        assert ball_row["z"][0] == pytest.approx(0.0)

        # Home GK (jersey 45) at first frame
        home_gk_row = first_frame.filter(pl.col("player_id") == "home_45")
        assert len(home_gk_row) == 1
        assert home_gk_row["x"][0] == pytest.approx(-47.78)
        assert home_gk_row["y"][0] == pytest.approx(0.59)

        # Away GK (jersey 1) at first frame
        away_gk_row = first_frame.filter(pl.col("player_id") == "away_1")
        assert len(away_gk_row) == 1
        assert away_gk_row["x"][0] == pytest.approx(46.48)
        assert away_gk_row["y"][0] == pytest.approx(0.23)

    def test_period_info(self):
        """Test period information."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            only_alive=False,
        )
        periods_df = dataset.periods

        # Should have 2 periods
        assert len(periods_df) == 2
        assert 1 in periods_df["period_id"].to_list()
        assert 2 in periods_df["period_id"].to_list()

    def test_timestamp_values(self):
        """Test timestamp values."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            only_alive=False,
        )
        tracking_df = dataset.tracking

        # First frame timestamp should be 12ms
        first_frame = tracking_df.filter(pl.col("frame_id") == 0)
        ball_row = first_frame.filter(pl.col("team_id") == "ball")
        # Timestamp is stored as timedelta, convert to milliseconds
        timestamp_ms = ball_row["timestamp"][0].total_seconds() * 1000
        assert timestamp_ms == 12  # milliseconds


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestSignalityLazy:
    """Test lazy loading."""

    def test_lazy_loading_basic(self):
        """Test basic lazy loading."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            lazy=True,
        )

        # Should be a LazyFrame
        assert isinstance(dataset.tracking, pl.LazyFrame)

        # Can collect
        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        # Default only_alive=True: 180 rows
        assert len(tracking_df) == 180

    def test_lazy_collect_with_filter(self):
        """Test lazy loading with filter pushdown."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            lazy=True,
            only_alive=False,
        )

        # Filter to period 1 only
        filtered = dataset.tracking.filter(pl.col("period_id") == 1).collect()
        assert all(filtered["period_id"] == 1)


class TestLazyNotImplemented:
    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            signality.load_tracking(
                meta_data=META_DATA,
                raw_data_feeds=RAW_DATA_FEEDS,
                venue_information=VENUE_INFO,
                lazy=True,
            )

    def test_from_cache_raises(self):
        with pytest.raises(NotImplementedError, match="cache loading"):
            signality.load_tracking(
                meta_data=META_DATA,
                raw_data_feeds=RAW_DATA_FEEDS,
                venue_information=VENUE_INFO,
                from_cache=True,
            )


class TestSignalityParallel:
    """Test parallel processing."""

    def test_parallel_true(self):
        """Test loading with parallel=True."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            parallel=True,
        )
        # Default only_alive=True: 180 rows
        assert len(dataset.tracking) == 180

    def test_parallel_false(self):
        """Test loading with parallel=False."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            parallel=False,
        )
        # Default only_alive=True: 180 rows
        assert len(dataset.tracking) == 180

    def test_parallel_results_consistent(self):
        """Test parallel and sequential produce same results."""
        dataset_par = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            parallel=True,
            only_alive=False,
        )
        dataset_seq = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            parallel=False,
            only_alive=False,
        )

        # Same number of rows
        assert len(dataset_par.tracking) == len(dataset_seq.tracking)

        # Same frame IDs
        par_frames = dataset_par.tracking["frame_id"].unique().sort()
        seq_frames = dataset_seq.tracking["frame_id"].unique().sort()
        assert par_frames.equals(seq_frames)


class TestSignalityOfficials:
    """Test officials support."""

    def test_include_officials_false(self):
        """Test officials excluded by default."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            include_officials=False,
            only_alive=False,
        )
        tracking_df = dataset.tracking
        team_df = dataset.teams

        # No officials team
        assert "officials" not in team_df["team_id"].to_list()

        # No officials in tracking
        officials_rows = tracking_df.filter(pl.col("team_id") == "officials")
        assert len(officials_rows) == 0

    def test_include_officials_true(self):
        """Test officials included when requested."""
        dataset = signality.load_tracking(
            meta_data=META_DATA,
            raw_data_feeds=RAW_DATA_FEEDS,
            venue_information=VENUE_INFO,
            include_officials=True,
            only_alive=False,
        )
        tracking_df = dataset.tracking
        team_df = dataset.teams

        # Officials team present
        assert "officials" in team_df["team_id"].to_list()

        # Officials in tracking
        # P1: 4 frames × 2 refs + 1 frame × 3 refs = 11
        # P2: 2 frames × 2 refs + 2 frames × 3 refs + 1 frame × 2 refs = 12
        # Total: 23 officials rows
        officials_rows = tracking_df.filter(pl.col("team_id") == "officials")
        assert len(officials_rows) == 23
