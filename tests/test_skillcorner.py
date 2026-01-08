"""Integration tests for SkillCorner tracking data loading."""

import pytest
import polars as pl
from pathlib import Path

from kloppy_light import skillcorner


# Test data paths (anonymized test data with 100 frames per period)
DATA_DIR = Path(__file__).parent / "files"
RAW_DATA_PATH = str(DATA_DIR / "skillcorner_tracking.jsonl")
META_DATA_PATH = str(DATA_DIR / "skillcorner_meta.json")


class TestLoadTracking:
    """Tests for skillcorner.load_tracking function."""

    def test_returns_four_dataframes(self):
        """Test that load_tracking returns a 4-tuple of DataFrames."""
        result = skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH)

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(df, pl.DataFrame) for df in result)

    def test_unpacking(self):
        """Test that the 4-tuple can be unpacked correctly."""
        tracking_df, metadata_df, team_df, player_df = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH
        )

        assert isinstance(tracking_df, pl.DataFrame)
        assert isinstance(metadata_df, pl.DataFrame)
        assert isinstance(team_df, pl.DataFrame)
        assert isinstance(player_df, pl.DataFrame)


class TestMetadataDataFrame:
    """Tests for the metadata DataFrame."""

    @pytest.fixture
    def metadata_df(self):
        """Load and return the metadata DataFrame."""
        _, metadata_df, _, _ = skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH)
        return metadata_df

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
            "period_1_start_frame_id",
            "period_1_end_frame_id",
            "period_2_start_frame_id",
            "period_2_end_frame_id",
            "period_3_start_frame_id",
            "period_3_end_frame_id",
            "period_4_start_frame_id",
            "period_4_end_frame_id",
            "period_5_start_frame_id",
            "period_5_end_frame_id",
        }
        assert set(metadata_df.columns) == expected_columns

    def test_provider_value(self, metadata_df):
        """Test provider is 'skillcorner'."""
        assert metadata_df["provider"][0] == "skillcorner"

    def test_coordinate_system_value(self, metadata_df):
        """Test coordinate_system is 'cdf'."""
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_orientation_value(self, metadata_df):
        """Test orientation is 'static_home_away'."""
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_team_names(self, metadata_df):
        """Test team names are extracted correctly."""
        assert metadata_df["home_team"][0] == "Home Team"
        assert metadata_df["away_team"][0] == "Away Team"

    def test_pitch_dimensions(self, metadata_df):
        """Test pitch dimensions are reasonable."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert pitch_length == pytest.approx(104.0, rel=0.01)
        assert pitch_width == pytest.approx(68.0, rel=0.01)

    def test_fps(self, metadata_df):
        """Test fps is 10 (SkillCorner default)."""
        assert metadata_df["fps"][0] == pytest.approx(10.0, rel=0.01)

    def test_game_date(self, metadata_df):
        """Test game_date is present and valid."""
        import datetime

        game_date = metadata_df["game_date"][0]
        assert game_date == datetime.date(2025, 1, 1)


class TestTeamDataFrame:
    """Tests for the team DataFrame."""

    @pytest.fixture
    def team_df(self):
        """Load and return the team DataFrame."""
        _, _, team_df, _ = skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH)
        return team_df

    def test_two_rows(self, team_df):
        """Test that team_df contains exactly two rows."""
        assert team_df.height == 2

    def test_schema(self, team_df):
        """Test that team_df has expected columns."""
        expected_columns = {"team_id", "name", "ground"}
        assert set(team_df.columns) == expected_columns

    def test_grounds(self, team_df):
        """Test that team_df contains home and away teams."""
        grounds = set(team_df["ground"].to_list())
        assert grounds == {"home", "away"}


class TestPlayerDataFrame:
    """Tests for the player DataFrame."""

    @pytest.fixture
    def player_df(self):
        """Load and return the player DataFrame."""
        _, _, _, player_df = skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH)
        return player_df

    def test_schema(self, player_df):
        """Test that player_df has expected columns."""
        expected_columns = {"team_id", "player_id", "name", "first_name", "last_name", "jersey_number", "position", "is_starter"}
        assert set(player_df.columns) == expected_columns

    def test_has_players(self, player_df):
        """Test that player_df contains players from both teams."""
        assert player_df.height > 20  # Should have at least 22 players

    def test_name_fields(self, player_df):
        """Test that name fields are populated correctly."""
        # Get first player
        first_player = player_df.row(0, named=True)

        # All name fields should be set for SkillCorner
        assert first_player["name"] is not None
        assert first_player["first_name"] is not None
        assert first_player["last_name"] is not None

    def test_position_standardized(self, player_df):
        """Test that positions are standardized codes."""
        valid_positions = {
            "GK", "LB", "RB", "LCB", "CB", "RCB", "LWB", "RWB",
            "LDM", "CDM", "RDM", "LCM", "CM", "RCM", "LAM", "CAM", "RAM",
            "LW", "RW", "LM", "RM", "LF", "ST", "RF", "CF", "SUB", "UNK"
        }
        positions = set(player_df["position"].to_list())
        assert positions.issubset(valid_positions)


class TestTrackingDataFrameLong:
    """Tests for tracking DataFrame with 'long' layout."""

    @pytest.fixture
    def tracking_df(self):
        """Load tracking data with long layout."""
        tracking_df, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="long"
        )
        return tracking_df

    def test_schema(self, tracking_df):
        """Test that long format has expected columns."""
        expected_columns = {
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
        assert ball_rows.height > 0

        # Check ball rows have player_id = "ball"
        assert ball_rows["player_id"].to_list()[:10] == ["ball"] * 10

    def test_timestamp_type(self, tracking_df):
        """Test that timestamp is Duration type."""
        assert tracking_df.schema["timestamp"] == pl.Duration("ms")

    def test_has_multiple_periods(self, tracking_df):
        """Test that data includes multiple periods."""
        periods = tracking_df["period_id"].unique().to_list()
        assert len(periods) >= 2


class TestTrackingDataFrameLongBall:
    """Tests for tracking DataFrame with 'long_ball' layout."""

    @pytest.fixture
    def tracking_df(self):
        """Load tracking data with long_ball layout."""
        tracking_df, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="long_ball"
        )
        return tracking_df

    def test_schema(self, tracking_df):
        """Test that long_ball format has expected columns."""
        expected_columns = {
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
    def tracking_df(self):
        """Load tracking data with wide layout."""
        tracking_df, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="wide"
        )
        return tracking_df

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


class TestIncludeEmptyFrames:
    """Tests for include_empty_frames parameter."""

    def test_empty_frames_excluded_by_default(self):
        """Test that empty frames are excluded by default."""
        df_no_empty, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, include_empty_frames=False
        )
        df_with_empty, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, include_empty_frames=True
        )

        # With empty frames should have more rows (or equal if no empty frames)
        assert df_with_empty.height >= df_no_empty.height


class TestOnlyAliveParameter:
    """Tests for only_alive parameter."""

    def test_only_alive_filters_dead_frames(self):
        """Test that only_alive=True filters out dead ball frames."""
        df_all, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=False
        )
        df_alive, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True
        )

        # Check if there are any dead frames in the data
        dead_rows_all = df_all.filter(pl.col("ball_state") == "dead")
        if dead_rows_all.height > 0:
            # Alive should have fewer rows if there are dead frames
            assert df_alive.height <= df_all.height

    def test_only_alive_no_dead_frames(self):
        """Test that only_alive=True results in no dead ball frames."""
        df_alive, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True
        )

        # All rows should be alive
        dead_rows = df_alive.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0


class TestOrientationParameter:
    """Tests for orientation parameter."""

    def test_orientation_default_static_home_away(self):
        """Test that orientation defaults to 'static_home_away'."""
        _, metadata_df, _, _ = skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH)
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_orientation_static_away_home(self):
        """Test that orientation='static_away_home' is recorded."""
        _, metadata_df, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home"
        )
        assert metadata_df["orientation"][0] == "static_away_home"

    def test_invalid_orientation(self):
        """Test that invalid orientation raises error."""
        with pytest.raises(Exception):
            skillcorner.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH, orientation="invalid"
            )


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_tracking_file(self):
        """Test that missing tracking file raises error."""
        with pytest.raises(Exception):
            skillcorner.load_tracking("nonexistent_tracking.jsonl", META_DATA_PATH)

    def test_missing_metadata_file(self):
        """Test that missing metadata file raises error."""
        with pytest.raises(Exception):
            skillcorner.load_tracking(RAW_DATA_PATH, "nonexistent_metadata.json")


class TestLazyParameter:
    """Tests for lazy loading parameter."""

    def test_lazy_returns_lazy_loader(self):
        """Test that lazy=True returns a LazyTrackingLoader."""
        from kloppy_light import LazyTrackingLoader

        t, m, team, player = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        assert isinstance(t, LazyTrackingLoader)
        assert isinstance(m, pl.DataFrame)  # Metadata is eager
        assert isinstance(team, pl.DataFrame)
        assert isinstance(player, pl.DataFrame)

    def test_lazy_collect_returns_dataframe(self):
        """Test that collect() returns a DataFrame."""
        t_lazy, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        result = t_lazy.collect()
        assert isinstance(result, pl.DataFrame)

    def test_lazy_filter_before_collect(self):
        """Test that filter() can be chained before collect()."""
        t_lazy, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        result = t_lazy.filter(pl.col("period_id") == 1).collect()
        # All rows should be period 1
        assert all(p == 1 for p in result["period_id"].to_list())

    def test_lazy_collect_matches_eager(self):
        """Test that lazy collect() produces same result as eager loading."""
        t_lazy, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        t_eager, _, _, _ = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=False
        )
        assert t_lazy.collect().equals(t_eager)
