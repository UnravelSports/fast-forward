"""Tests for Respovision provider."""

import pytest
from datetime import date

import polars as pl

from fastforward import respovision
from tests.config import RV_RAW as RAW_DATA_SIMPLE

# Test data summary:
# - 8 frames total: 6 in half_1 (frames 1-6), 2 in half_2 (frames 101-102)
# - 22 players (11 per team) + 1 referee per frame
# - Frames 4-5 have ball_possession=null (dead ball)
# - Frame 6 has null ball coordinates (missing ball tracking)
# - Pitch dimensions: 100x64
# - Teams: "Home Team" (right side half_1), "Away Team" (left side half_1)
# - With default exclude_missing_ball_frames=True, frame 6 is excluded -> 7 frames


class TestRespovisionBasic:
    """Basic Respovision loading tests."""

    def test_load_basic(self):
        """Test basic loading."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )

        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)
        assert isinstance(dataset.periods, pl.DataFrame)

    def test_tracking_row_count(self):
        """Test tracking DataFrame row count.

        Test data: 7 frames × (22 players + 1 ball) = 161 rows
        """
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
            include_officials=False,
        )

        # 7 frames × 23 entities (22 players + 1 ball) = 161 rows
        assert len(dataset.tracking) == 161

    def test_tracking_row_count_with_officials(self):
        """Test tracking DataFrame row count with officials.

        Test data: 7 frames × (22 players + 1 ball + 1 referee) = 168 rows
        """
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
            include_officials=True,
        )

        # 7 frames × 24 entities (22 players + 1 ball + 1 referee) = 168 rows
        assert len(dataset.tracking) == 168

    def test_metadata_row_count(self):
        """Test metadata DataFrame has exactly 1 row."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )

        assert len(dataset.metadata) == 1

    def test_teams_row_count(self):
        """Test teams DataFrame has 2 rows (home and away)."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )

        assert len(dataset.teams) == 2

    def test_teams_row_count_with_officials(self):
        """Test teams DataFrame has 3 rows when including officials."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=True,
        )

        # Home, Away, Officials
        assert len(dataset.teams) == 3

    def test_players_row_count(self):
        """Test players DataFrame has 22 players (11 per team)."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )

        assert len(dataset.players) == 22

    def test_players_row_count_with_officials(self):
        """Test players DataFrame has 23 entries when including officials."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=True,
        )

        # 22 players + 1 referee
        assert len(dataset.players) == 23

    def test_periods_row_count(self):
        """Test periods DataFrame has 2 rows (half_1 and half_2)."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )

        assert len(dataset.periods) == 2

    def test_lazy_not_supported(self):
        """Test that lazy=True raises ValueError."""
        with pytest.raises(ValueError, match="lazy=True is not supported"):
            respovision.load_tracking(RAW_DATA_SIMPLE, lazy=True)


class TestRespovisionColumns:
    """Test DataFrame column structure."""

    def test_tracking_columns_with_angles(self):
        """Test tracking_df has expected columns including angles."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_joint_angles=True,
        )
        tracking_df = dataset.tracking

        expected_cols = [
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
            "head_angle",
            "shoulders_angle",
            "hips_angle",
        ]
        assert all(col in tracking_df.columns for col in expected_cols)

    def test_tracking_columns_without_angles(self):
        """Test tracking_df columns without angles."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_joint_angles=False,
        )
        tracking_df = dataset.tracking

        # Should have base columns but not angle columns
        assert "x" in tracking_df.columns
        assert "y" in tracking_df.columns
        assert "z" in tracking_df.columns
        assert "head_angle" not in tracking_df.columns
        assert "shoulders_angle" not in tracking_df.columns
        assert "hips_angle" not in tracking_df.columns

    def test_metadata_columns(self):
        """Test metadata_df has expected columns."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        metadata_df = dataset.metadata

        expected_cols = [
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
        ]
        assert all(col in metadata_df.columns for col in expected_cols)

    def test_team_columns(self):
        """Test team_df has expected columns."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        team_df = dataset.teams

        expected_cols = ["game_id", "team_id", "name", "ground"]
        assert all(col in team_df.columns for col in expected_cols)

    def test_player_columns(self):
        """Test player_df has expected columns."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        player_df = dataset.players

        expected_cols = [
            "game_id",
            "team_id",
            "player_id",
            "name",
            "jersey_number",
            "position",
        ]
        assert all(col in player_df.columns for col in expected_cols)

    def test_periods_columns(self):
        """Test periods_df has expected columns."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )
        periods_df = dataset.periods

        expected_cols = ["game_id", "period_id", "start_frame_id", "end_frame_id", "start_timestamp", "end_timestamp", "duration"]
        assert all(col in periods_df.columns for col in expected_cols)


class TestRespovisionMetadataValues:
    """Test metadata values."""

    def test_provider_value(self):
        """Test provider is 'respovision'."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        assert dataset.metadata["provider"][0] == "respovision"

    def test_pitch_dimensions(self):
        """Test pitch dimensions are set correctly."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        assert dataset.metadata["pitch_length"][0] == pytest.approx(100.0)
        assert dataset.metadata["pitch_width"][0] == pytest.approx(64.0)

    def test_pitch_dimensions_custom(self):
        """Test custom pitch dimensions."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=105.0,
            pitch_width=68.0,
        )
        assert dataset.metadata["pitch_length"][0] == pytest.approx(105.0)
        assert dataset.metadata["pitch_width"][0] == pytest.approx(68.0)

    def test_fps_value(self):
        """Test FPS value (25 Hz based on 40ms intervals)."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        assert dataset.metadata["fps"][0] == pytest.approx(25.0)

    def test_coordinate_system_default(self):
        """Test default coordinate system is 'cdf'."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        assert dataset.metadata["coordinate_system"][0] == "cdf"

    def test_orientation_default(self):
        """Test default orientation is 'static_home_away'."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        assert dataset.metadata["orientation"][0] == "static_home_away"


class TestRespovisionTeamValues:
    """Test team values."""

    def test_team_ids(self):
        """Test team IDs are 'home' and 'away'."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )
        team_ids = sorted(dataset.teams["team_id"].to_list())
        assert team_ids == ["away", "home"]

    def test_team_names(self):
        """Test team names match original."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )
        team_names = sorted(dataset.teams["name"].to_list())
        assert team_names == ["Away Team", "Home Team"]

    def test_team_grounds(self):
        """Test team grounds are 'home' and 'away'."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )
        grounds = sorted(dataset.teams["ground"].to_list())
        assert grounds == ["away", "home"]


class TestRespovisionPlayerValues:
    """Test player values."""

    def test_player_id_format(self):
        """Test player IDs are formatted as {home|away}_jersey."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )
        player_ids = dataset.players["player_id"].to_list()

        # All player IDs should contain underscore and be lowercase
        for pid in player_ids:
            assert "_" in pid
            parts = pid.split("_")
            assert len(parts) == 2
            assert parts[0] in ("home", "away")
            assert parts[1].isdigit()
            assert pid == pid.lower()

    def test_specific_player_ids(self):
        """Test specific player IDs."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )
        player_ids = set(dataset.players["player_id"].to_list())

        # Check some expected player IDs (home_N or away_N format)
        assert "home_1" in player_ids  # Goalkeeper
        assert "home_10" in player_ids
        assert "away_1" in player_ids  # Goalkeeper
        assert "away_10" in player_ids

    def test_jersey_numbers(self):
        """Test jersey numbers are integers 1-11."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )
        jersey_numbers = set(dataset.players["jersey_number"].to_list())

        # Both teams have jerseys 1-11
        expected = set(range(1, 12))
        assert jersey_numbers == expected

    def test_goalkeeper_position(self):
        """Test goalkeepers have position 'GK'."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
        )
        goalkeepers = dataset.players.filter(pl.col("jersey_number") == 1)

        # Both team goalkeepers should have GK position
        assert len(goalkeepers) == 2
        for pos in goalkeepers["position"].to_list():
            assert pos == "GK"

    def test_is_starter_column(self):
        """Test is_starter column is populated correctly."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
            only_alive=False,
        )

        # is_starter column should exist
        assert "is_starter" in dataset.players.columns

        # All players in test data are in the first frame (all starters)
        starters = dataset.players.filter(pl.col("is_starter") == True)
        assert len(starters) == 22  # All 22 players are starters


class TestRespovisionPeriodValues:
    """Test period values."""

    def test_period_ids(self):
        """Test period IDs are 1 and 2."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )
        period_ids = sorted(dataset.periods["period_id"].to_list())
        assert period_ids == [1, 2]

    def test_period_frame_boundaries(self):
        """Test period frame boundaries."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )
        periods_df = dataset.periods.sort("period_id")

        # Period 1: frames 1-5
        p1 = periods_df.filter(pl.col("period_id") == 1)
        assert p1["start_frame_id"][0] == 1
        assert p1["end_frame_id"][0] == 5

        # Period 2: frames 101-102
        p2 = periods_df.filter(pl.col("period_id") == 2)
        assert p2["start_frame_id"][0] == 101
        assert p2["end_frame_id"][0] == 102

    def test_period_timing(self):
        """Test that all periods have correct timing values."""
        from datetime import timedelta

        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )
        periods = dataset.periods.sort("period_id")

        # Period 1: 5 frames at 25fps (0, 40, 80, 120, 160ms)
        p1 = periods.row(0, named=True)
        assert p1["start_timestamp"] == timedelta(milliseconds=0)
        assert p1["end_timestamp"] == timedelta(milliseconds=160)
        assert p1["duration"] == timedelta(milliseconds=160)

        # Period 2: 2 frames at 25fps (0, 40ms)
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] == timedelta(milliseconds=0)
        assert p2["end_timestamp"] == timedelta(milliseconds=40)
        assert p2["duration"] == timedelta(milliseconds=40)


class TestRespovisionTrackingValues:
    """Test tracking data values."""

    def test_ball_state_values(self):
        """Test ball_state values are 'alive' or 'dead'."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )
        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball")
        ball_states = set(ball_rows["ball_state"].to_list())

        # Frames 1-3, 101-102 have possession (alive), frames 4-5 have null (dead)
        assert ball_states == {"alive", "dead"}

    def test_only_alive_filter(self):
        """Test only_alive=True filters dead ball frames."""
        dataset_all = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )
        dataset_alive = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=True,
        )

        # All: 7 frames × 23 = 161 rows
        # Alive: 5 frames × 23 = 115 rows (frames 1-3 and 101-102)
        assert len(dataset_all.tracking) == 161
        assert len(dataset_alive.tracking) == 115

        # All ball rows in alive should have ball_state == "alive"
        ball_rows = dataset_alive.tracking.filter(pl.col("team_id") == "ball")
        assert all(ball_rows["ball_state"] == "alive")

    def test_ball_owning_team_id(self):
        """Test ball_owning_team_id values."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )

        # Frames 1-3: Home Team possession -> "home"
        # Frames 4-5: null (dead ball)
        # Frames 101-102: Away Team possession -> "away"
        alive_frames = dataset.tracking.filter(pl.col("ball_state") == "alive")
        owning_teams = alive_frames["ball_owning_team_id"].unique().to_list()

        assert "home" in owning_teams
        assert "away" in owning_teams

    def test_timestamp_type(self):
        """Test timestamp is Duration type."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )
        assert dataset.tracking.schema["timestamp"] == pl.Duration("ms")

    def test_timestamp_values(self):
        """Test timestamp values are correct (period-relative)."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )

        # Get first frame of each period (ball rows)
        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball").sort("frame_id")

        # Frame 1 (period 1): timestamp = 0ms
        frame1 = ball_rows.filter(pl.col("frame_id") == 1)
        assert frame1["timestamp"][0].total_seconds() == pytest.approx(0.0)

        # Frame 2 (period 1): timestamp = 40ms
        frame2 = ball_rows.filter(pl.col("frame_id") == 2)
        assert frame2["timestamp"][0].total_seconds() == pytest.approx(0.04)

        # Frame 101 (period 2): timestamp = 0ms (reset for new period)
        frame101 = ball_rows.filter(pl.col("frame_id") == 101)
        assert frame101["timestamp"][0].total_seconds() == pytest.approx(0.0)


class TestRespovisionCoordinates:
    """Test coordinate transformation."""

    def test_cdf_coordinates_center_origin(self):
        """Test CDF coordinates have center origin."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            coordinates="cdf",
            only_alive=False,
        )

        # In CDF, center is (0, 0)
        # Ball in frame 1 is at native (50, 32) -> CDF (0, 0)
        ball_frame1 = dataset.tracking.filter(
            (pl.col("team_id") == "ball") & (pl.col("frame_id") == 1)
        )
        assert ball_frame1["x"][0] == pytest.approx(0.0, abs=0.5)
        assert ball_frame1["y"][0] == pytest.approx(0.0, abs=0.5)

    def test_respovision_native_coordinates(self):
        """Test native Respovision coordinates (bottom-left origin)."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            coordinates="respovision",
            only_alive=False,
        )

        # In native coords, ball in frame 1 should be at ~(50, 32)
        ball_frame1 = dataset.tracking.filter(
            (pl.col("team_id") == "ball") & (pl.col("frame_id") == 1)
        )
        assert ball_frame1["x"][0] == pytest.approx(50.0, abs=0.5)
        assert ball_frame1["y"][0] == pytest.approx(32.0, abs=0.5)

    def test_goalkeeper_positions_cdf(self):
        """Test goalkeeper positions in CDF coordinates with static_home_away orientation.

        With static_home_away orientation, home team always attacks right (+x).
        In the test data, home team starts on the right side (raw GK x=95),
        so coordinates are flipped to put home on the left (attacking right).

        After orientation transformation:
        - Home GK: defending left goal -> x=-45 (negative, left side)
        - Away GK: defending right goal -> x=+45 (positive, right side)
        """
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            coordinates="cdf",
            only_alive=False,
        )

        frame1 = dataset.tracking.filter(pl.col("frame_id") == 1)

        # Home GK defends left goal (negative x) when home attacks right
        home_gk = frame1.filter(pl.col("player_id") == "home_1")
        assert home_gk["x"][0] == pytest.approx(-45.0, abs=1.0)

        # Away GK defends right goal (positive x)
        away_gk = frame1.filter(pl.col("player_id") == "away_1")
        assert away_gk["x"][0] == pytest.approx(45.0, abs=1.0)


class TestRespovisionJointAngles:
    """Test joint angle handling."""

    def test_angles_included_by_default(self):
        """Test joint angles are included by default."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
        )

        assert "head_angle" in dataset.tracking.columns
        assert "shoulders_angle" in dataset.tracking.columns
        assert "hips_angle" in dataset.tracking.columns

    def test_angles_excluded_when_disabled(self):
        """Test joint angles excluded when include_joint_angles=False."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_joint_angles=False,
        )

        assert "head_angle" not in dataset.tracking.columns
        assert "shoulders_angle" not in dataset.tracking.columns
        assert "hips_angle" not in dataset.tracking.columns

    def test_goalkeeper_angles_are_null(self):
        """Test goalkeeper angles are null (NaN in source)."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_joint_angles=True,
            only_alive=False,
        )

        # Goalkeepers have NaN angles
        gk_rows = dataset.tracking.filter(
            (pl.col("player_id") == "home_1") | (pl.col("player_id") == "away_1")
        )

        # All goalkeeper head_angle values should be null
        assert gk_rows["head_angle"].is_null().all()
        assert gk_rows["shoulders_angle"].is_null().all()
        assert gk_rows["hips_angle"].is_null().all()

    def test_player_angles_are_valid(self):
        """Test non-goalkeeper players have valid angles."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_joint_angles=True,
            only_alive=False,
        )

        # Get non-goalkeeper, non-ball rows
        player_rows = dataset.tracking.filter(
            (pl.col("team_id") != "ball")
            & (pl.col("player_id") != "home_1")
            & (pl.col("player_id") != "away_1")
        )

        # Most players should have non-null angles
        non_null_head = player_rows.filter(pl.col("head_angle").is_not_null())
        assert len(non_null_head) > 0


class TestRespovisionOfficials:
    """Test officials/referee handling."""

    def test_officials_excluded_by_default(self):
        """Test officials are excluded by default."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=False,
            only_alive=False,
        )

        team_ids = set(dataset.tracking["team_id"].unique().to_list())
        assert "officials" not in team_ids

    def test_officials_included_when_enabled(self):
        """Test officials are included when include_officials=True."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=True,
            only_alive=False,
        )

        team_ids = set(dataset.tracking["team_id"].unique().to_list())
        assert "officials" in team_ids

    def test_referee_position_code(self):
        """Test referee has position 'REF'."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=True,
        )

        officials = dataset.players.filter(pl.col("team_id") == "officials")
        assert len(officials) == 1
        assert officials["position"][0] == "REF"

    def test_referee_id_format(self):
        """Test referee ID uses official_{person_id} format."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_officials=True,
        )

        officials = dataset.players.filter(pl.col("team_id") == "officials")
        assert len(officials) == 1
        # person_id=100 in test data
        assert officials["player_id"][0] == "official_100"


class TestRespovisionGameId:
    """Test game_id handling."""

    def test_include_game_id_true(self):
        """Test include_game_id=True adds game_id column."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_game_id=True,
        )

        assert "game_id" in dataset.tracking.columns
        assert "game_id" in dataset.teams.columns
        assert "game_id" in dataset.players.columns
        assert "game_id" in dataset.periods.columns

    def test_include_game_id_false(self):
        """Test include_game_id=False omits game_id column."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_game_id=False,
        )

        assert "game_id" not in dataset.tracking.columns
        assert "game_id" not in dataset.teams.columns
        assert "game_id" not in dataset.players.columns

    def test_include_game_id_custom_string(self):
        """Test include_game_id with custom string."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            include_game_id="my-custom-game-id",
        )

        assert dataset.tracking["game_id"][0] == "my-custom-game-id"
        assert dataset.metadata["game_id"][0] == "my-custom-game-id"


class TestRespovisionLayouts:
    """Test different layout options."""

    def test_long_layout(self):
        """Test long layout has ball as separate rows."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            layout="long",
            only_alive=False,
        )

        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball")
        assert len(ball_rows) == 7  # One ball row per frame

    def test_long_ball_layout(self):
        """Test long_ball layout has ball as columns."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            layout="long_ball",
            only_alive=False,
        )

        # No ball rows in long_ball
        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball")
        assert len(ball_rows) == 0

        # Should have ball_x, ball_y, ball_z columns
        assert "ball_x" in dataset.tracking.columns
        assert "ball_y" in dataset.tracking.columns
        assert "ball_z" in dataset.tracking.columns

    def test_wide_layout(self):
        """Test wide layout has one row per frame."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            layout="wide",
            only_alive=False,
        )

        # Wide layout: one row per frame
        assert len(dataset.tracking) == 7  # 7 frames total

        # Should have player columns like "home_10_x"
        columns = dataset.tracking.columns
        player_cols = [c for c in columns if "_x" in c and "ball" not in c.lower()]
        assert len(player_cols) > 0

        # Check specific column names
        assert "home_10_x" in columns
        assert "away_10_x" in columns


class TestRespovisionExcludeMissingBallFrames:
    """Tests for exclude_missing_ball_frames parameter."""

    def test_exclude_missing_ball_frames_default_true(self):
        """Test that exclude_missing_ball_frames defaults to True.

        Test data has 8 frames total, 1 has missing ball (frame 6).
        With default exclude_missing_ball_frames=True, frame 6 is excluded.
        7 frames × 23 entities = 161 rows.
        """
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
        )
        # 8 total frames, 1 has missing ball -> 7 frames included
        # 7 frames × 23 entities = 161 rows
        assert len(dataset.tracking) == 161

    def test_exclude_missing_ball_frames_false_includes_all(self):
        """Test that exclude_missing_ball_frames=False includes frames with missing ball."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
            exclude_missing_ball_frames=False,
        )
        # 8 frames × 23 entities = 184 rows
        assert len(dataset.tracking) == 184

    def test_missing_ball_frame_has_nan_coordinates(self):
        """Test that missing ball frames have NaN coordinates when not excluded."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
            exclude_missing_ball_frames=False,
        )
        # Frame 6 has missing ball
        ball_frame6 = dataset.tracking.filter(
            (pl.col("team_id") == "ball") & (pl.col("frame_id") == 6)
        )
        assert len(ball_frame6) == 1
        assert ball_frame6["x"].is_nan()[0]
        assert ball_frame6["y"].is_nan()[0]
        assert ball_frame6["z"].is_nan()[0]

    def test_exclude_missing_ball_frames_filters_correct_count(self):
        """Test exact filtering: 8 frames total, 1 missing ball = 7 included."""
        dataset_all = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
            exclude_missing_ball_frames=False,
        )
        dataset_filtered = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
            exclude_missing_ball_frames=True,
        )
        # All: 8 frames × 23 = 184 rows
        # Filtered: 7 frames × 23 = 161 rows (frame 6 excluded)
        assert len(dataset_all.tracking) == 184
        assert len(dataset_filtered.tracking) == 161

    def test_missing_ball_frame_excluded_from_frame_ids(self):
        """Test that frame 6 is not present when exclude_missing_ball_frames=True."""
        dataset = respovision.load_tracking(
            RAW_DATA_SIMPLE,
            pitch_length=100.0,
            pitch_width=64.0,
            only_alive=False,
            exclude_missing_ball_frames=True,
        )
        frame_ids = set(dataset.tracking["frame_id"].unique().to_list())
        assert 6 not in frame_ids
        # Other frames should be present
        assert 1 in frame_ids
        assert 5 in frame_ids
        assert 101 in frame_ids


class TestRespovisionErrorHandling:
    """Test error handling."""

    def test_missing_file(self):
        """Test error on missing file."""
        with pytest.raises(Exception):
            respovision.load_tracking("nonexistent_file.jsonl")

    def test_invalid_layout(self):
        """Test error on invalid layout."""
        with pytest.raises(Exception):
            respovision.load_tracking(
                RAW_DATA_SIMPLE,
                pitch_length=100.0,
                pitch_width=64.0,
                layout="invalid",
            )

    def test_invalid_coordinates(self):
        """Test error on invalid coordinates."""
        with pytest.raises(Exception):
            respovision.load_tracking(
                RAW_DATA_SIMPLE,
                pitch_length=100.0,
                pitch_width=64.0,
                coordinates="invalid",
            )
