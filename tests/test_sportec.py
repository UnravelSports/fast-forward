"""Integration tests for Sportec tracking data loading."""

import pytest
import polars as pl

from fastforward import sportec
from tests.config import (
    SP_META as META_DATA_PATH,
    SP_META_BOM as META_DATA_BOM_PATH,
    SP_RAW as RAW_DATA_PATH,
    SP_RAW_BOM as RAW_DATA_BOM_PATH,
    SP_RAW_W_REF as RAW_DATA_W_REF_PATH,
)


class TestLoadTracking:
    """Tests for sportec.load_tracking function."""

    def test_returns_dataset(self):
        """Test that load_tracking returns a TrackingDataset."""
        from fastforward import TrackingDataset

        dataset = sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)


class TestMetadataDataFrame:
    """Tests for the metadata DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def metadata_df(self, dataset):
        """Return the metadata DataFrame."""
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
        """Test provider is 'sportec'."""
        assert metadata_df["provider"][0] == "sportec"

    def test_coordinate_system_value(self, metadata_df):
        """Test coordinate_system is 'cdf'."""
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_orientation_value(self, metadata_df):
        """Test orientation is 'static_home_away'."""
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_team_names(self, metadata_df):
        """Test team names are extracted correctly."""
        assert metadata_df["home_team"][0] == "Sport-Club Freiburg"
        assert metadata_df["away_team"][0] == "Borussia Mönchengladbach"

    def test_pitch_dimensions(self, metadata_df):
        """Test pitch dimensions are correct."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert pitch_length == pytest.approx(100.0, rel=0.01)
        assert pitch_width == pytest.approx(68.0, rel=0.01)

    def test_fps(self, metadata_df):
        """Test fps is 25 (Sportec default)."""
        assert metadata_df["fps"][0] == pytest.approx(25.0, rel=0.01)

    def test_game_id(self, metadata_df):
        """Test game_id is extracted correctly."""
        assert metadata_df["game_id"][0] == "DFL-MAT-003BN1"

    def test_game_date(self, metadata_df):
        """Test game_date is present and valid."""
        import datetime

        game_date = metadata_df["game_date"][0]
        assert game_date == datetime.date(2020, 6, 5)


class TestTeamDataFrame:
    """Tests for the team DataFrame."""

    @pytest.fixture
    def team_df(self):
        """Load and return the team DataFrame."""
        dataset = sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        team_df = dataset.teams
        return team_df

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


class TestPlayerDataFrame:
    """Tests for the player DataFrame."""

    @pytest.fixture
    def player_df(self):
        """Load and return the player DataFrame."""
        dataset = sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        player_df = dataset.players
        return player_df

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
        # Expected: 20 home + 20 away = 40 players
        assert player_df.height == 40

    def test_name_fields(self, player_df):
        """Test that name fields are populated correctly."""
        # Get first player
        first_player = player_df.row(0, named=True)

        # Sportec provides first_name and last_name
        assert first_player["first_name"] is not None
        assert first_player["last_name"] is not None

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


class TestPeriodsDataFrame:
    """Tests for the periods DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

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
        assert p1["start_timestamp"] == timedelta(milliseconds=0)
        assert p1["end_timestamp"] == timedelta(milliseconds=3960)
        assert p1["duration"] == timedelta(milliseconds=3960)

        # Period 2
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] == timedelta(milliseconds=0)
        assert p2["end_timestamp"] == timedelta(milliseconds=3920)
        assert p2["duration"] == timedelta(milliseconds=3920)


class TestTrackingDataFrameLong:
    """Tests for tracking DataFrame with 'long' layout."""

    @pytest.fixture
    def dataset(self):
        """Load dataset with long layout."""
        return sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="long", lazy=False
        )

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return the tracking DataFrame."""
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
        assert ball_rows.height == 199

        # Check ball rows have player_id = "ball"
        assert ball_rows["player_id"].to_list()[:10] == ["ball"] * min(
            10, ball_rows.height
        )

    def test_timestamp_type(self, tracking_df):
        """Test that timestamp is Duration type."""
        assert tracking_df.schema["timestamp"] == pl.Duration("ms")

    def test_has_multiple_periods(self, tracking_df):
        """Test that data includes multiple periods."""
        periods = sorted(tracking_df["period_id"].unique().to_list())
        assert periods == [1, 2]


class TestTrackingDataFrameLongBall:
    """Tests for tracking DataFrame with 'long_ball' layout."""

    @pytest.fixture
    def dataset(self):
        """Load dataset with long_ball layout."""
        return sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="long_ball", lazy=False
        )

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return the tracking DataFrame."""
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
        """Load dataset with wide layout."""
        return sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="wide", lazy=False
        )

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return the tracking DataFrame."""
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

        assert len(x_columns) == 3
        assert len(x_columns) == len(y_columns) == len(z_columns)

    def test_one_row_per_frame(self, tracking_df):
        """Test that wide format has exactly one row per frame."""
        frame_count = tracking_df["frame_id"].n_unique()
        assert tracking_df.height == frame_count


class TestOnlyAliveParameter:
    """Tests for only_alive parameter."""

    def test_only_alive_filters_dead_frames(self):
        """Test that only_alive=True filters out dead ball frames."""
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=False, lazy=False
        )
        df_all = dataset.tracking
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True, lazy=False
        )
        df_alive = dataset.tracking

        # alive should have fewer rows than all (some dead frames removed)
        assert df_all.height == 493
        assert df_alive.height == 481

    def test_only_alive_no_dead_frames(self):
        """Test that only_alive=True results in no dead ball frames."""
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True, lazy=False
        )
        df_alive = dataset.tracking

        # All rows should be alive
        dead_rows = df_alive.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0


class TestOrientationParameter:
    """Tests for orientation parameter."""

    def test_orientation_default_static_home_away(self):
        """Test that orientation defaults to 'static_home_away'."""
        dataset = sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        metadata_df = dataset.metadata
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_orientation_static_away_home(self):
        """Test that orientation='static_away_home' is recorded."""
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", lazy=False
        )
        metadata_df = dataset.metadata
        assert metadata_df["orientation"][0] == "static_away_home"

    def test_invalid_orientation(self):
        """Test that invalid orientation raises error."""
        with pytest.raises(Exception):
            sportec.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH, orientation="invalid", lazy=False
            )


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_tracking_file(self):
        """Test that missing tracking file raises error."""
        with pytest.raises(Exception):
            sportec.load_tracking("nonexistent_tracking.xml", META_DATA_PATH, lazy=False)

    def test_missing_metadata_file(self):
        """Test that missing metadata file raises error."""
        with pytest.raises(Exception):
            sportec.load_tracking(RAW_DATA_PATH, "nonexistent_metadata.xml", lazy=False)


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyParameter:
    """Tests for lazy loading parameter."""

    def test_lazy_returns_lazyframe(self):
        """Test that lazy=True returns a pl.LazyFrame."""
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        assert isinstance(dataset.tracking, pl.LazyFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)  # Metadata is eager
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)

    def test_lazy_collect_returns_dataframe(self):
        """Test that collect() returns a DataFrame."""
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        t_lazy = dataset.tracking
        result = t_lazy.collect()
        assert isinstance(result, pl.DataFrame)

    def test_lazy_filter_before_collect(self):
        """Test that filter() can be chained before collect()."""
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        t_lazy = dataset.tracking
        result = t_lazy.filter(pl.col("period_id") == 1).collect()
        # All rows should be period 1
        assert all(p == 1 for p in result["period_id"].to_list())

    def test_lazy_collect_matches_eager(self):
        """Test that lazy collect() produces same result as eager loading."""
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        t_lazy = dataset.tracking
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=False
        )
        t_eager = dataset.tracking
        assert t_lazy.collect().equals(t_eager)


class TestLazyNotImplemented:
    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

    def test_from_cache_raises(self):
        with pytest.raises(NotImplementedError, match="cache loading"):
            sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, from_cache=True)


class TestIncludeOfficials:
    """Tests for include_officials parameter (Sportec-specific)."""

    def test_officials_excluded_by_default(self):
        """Test that officials are excluded by default."""
        dataset = sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        player_df = dataset.players

        # Should have 40 players (20 per team, no officials)
        assert player_df.height == 40

        # No officials team_id
        officials_rows = player_df.filter(pl.col("team_id") == "officials")
        assert officials_rows.height == 0

    def test_officials_included_when_enabled(self):
        """Test that officials are included when include_officials=True."""
        dataset = sportec.load_tracking(
            RAW_DATA_W_REF_PATH, META_DATA_PATH, include_officials=True, lazy=False
        )
        player_df = dataset.players

        # Should have 44 rows: 40 players + 4 officials
        assert player_df.height == 44

        # Should have officials rows
        officials_rows = player_df.filter(pl.col("team_id") == "officials")
        assert officials_rows.height == 4

    def test_officials_team_id_is_officials(self):
        """Test that officials have team_id = 'officials'."""
        dataset = sportec.load_tracking(
            RAW_DATA_W_REF_PATH, META_DATA_PATH, include_officials=True, lazy=False
        )
        player_df = dataset.players

        officials_rows = player_df.filter(pl.col("team_id") == "officials")
        assert all(t == "officials" for t in officials_rows["team_id"].to_list())

    def test_officials_positions(self):
        """Test that officials have correct position codes."""
        dataset = sportec.load_tracking(
            RAW_DATA_W_REF_PATH, META_DATA_PATH, include_officials=True, lazy=False
        )
        player_df = dataset.players

        officials_rows = player_df.filter(pl.col("team_id") == "officials")
        positions = set(officials_rows["position"].to_list())

        # Should have REF, AREF (2x), and 4TH
        expected_positions = {"REF", "AREF", "4TH"}
        assert positions.issubset(expected_positions | {"UNK"})
        assert "REF" in positions  # Main official must be present


class TestSpecificValues:
    """Tests for specific data values matching kloppy test expectations."""

    @pytest.fixture
    def dataset_wide(self):
        """Load dataset with wide layout."""
        return sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="wide", only_alive=False, lazy=False
        )

    @pytest.fixture
    def tracking_wide_all(self, dataset_wide):
        """Return the tracking DataFrame."""
        return dataset_wide.tracking

    @pytest.fixture
    def dataset_long(self):
        """Load dataset with long layout."""
        return sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="long", only_alive=False, lazy=False
        )

    @pytest.fixture
    def tracking_long_all(self, dataset_long):
        """Return the tracking DataFrame."""
        return dataset_long.tracking

    def test_total_frame_count_all(self, tracking_wide_all):
        """Test frame count with only_alive=False.

        The test data has 205 unique frames (101 period 1 + 104 period 2).
        """
        assert tracking_wide_all.height == 205

    def test_total_frame_count_default(self):
        """Test frame count with default only_alive=True.

        With only_alive=True (default), should have 199 frames (6 dead frames removed).
        """
        dataset = sportec.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="wide", lazy=False
        )
        tracking_df = dataset.tracking
        assert tracking_df.height == 199

    def test_period_1_kickoff_frame(self, tracking_wide_all):
        """Test that period 1 starts with frame 10000."""
        period_1 = tracking_wide_all.filter(pl.col("period_id") == 1)
        assert period_1["frame_id"][0] == 10000

    def test_period_2_kickoff_frame(self, tracking_wide_all):
        """Test that period 2 starts with frame 100000."""
        period_2 = tracking_wide_all.filter(pl.col("period_id") == 2)
        assert period_2["frame_id"][0] == 100000

    def test_ball_coordinates_frame_10000(self, tracking_long_all):
        """Test ball coordinates at frame 10000.

        Expected: Point3D(x=2.69, y=0.26, z=0.06)
        Note: Frame 10000 is a dead ball frame, need only_alive=False.
        """
        ball_10000 = tracking_long_all.filter(
            (pl.col("frame_id") == 10000) & (pl.col("team_id") == "ball")
        )
        assert ball_10000.height == 1
        assert ball_10000["x"][0] == pytest.approx(2.69, abs=0.01)
        assert ball_10000["y"][0] == pytest.approx(0.26, abs=0.01)
        assert ball_10000["z"][0] == pytest.approx(0.06, abs=0.01)

    def test_ball_state_frame_10000(self, tracking_long_all):
        """Test that frame 10000 has dead ball state (kick-off setup)."""
        ball_10000 = tracking_long_all.filter(
            (pl.col("frame_id") == 10000) & (pl.col("team_id") == "ball")
        )
        assert ball_10000["ball_state"][0] == "dead"

    def test_player_coordinates_frame_10000(self, tracking_long_all):
        """Test player DFL-OBJ-002G3I coordinates at frame 10000.

        Expected: Point(x=0.35, y=-25.26)
        Note: Frame 10000 is a dead ball frame, need only_alive=False.
        """
        player_10000 = tracking_long_all.filter(
            (pl.col("frame_id") == 10000) & (pl.col("player_id") == "DFL-OBJ-002G3I")
        )
        assert player_10000.height == 1
        assert player_10000["x"][0] == pytest.approx(0.35, abs=0.01)
        assert player_10000["y"][0] == pytest.approx(-25.26, abs=0.01)

    def test_players_in_frame(self, tracking_long_all):
        """Test that frames contain expected number of tracked players.

        Frame 10000 has ball + 1 player (DFL-OBJ-002G3I) tracked.
        Other players appear in later frames.
        """
        frame_10000 = tracking_long_all.filter(pl.col("frame_id") == 10000)
        # Ball + 1 player in frame 10000
        assert frame_10000.height == 2

    def test_period_frame_ranges(self, tracking_wide_all):
        """Test the frame ID ranges for each period."""
        period_1 = tracking_wide_all.filter(pl.col("period_id") == 1)
        period_2 = tracking_wide_all.filter(pl.col("period_id") == 2)

        # Period 1: frames 10000-10100
        assert period_1["frame_id"].min() == 10000
        assert period_1["frame_id"].max() == 10100
        assert period_1.height == 101

        # Period 2: frames 100000-100103
        assert period_2["frame_id"].min() == 100000
        assert period_2["frame_id"].max() == 100103
        assert period_2.height == 104

    def test_player_first_appearance(self, tracking_long_all):
        """Test when player DFL-OBJ-002G5S first appears.

        This player appears first in the 27th frame.
        """
        player_frames = tracking_long_all.filter(
            pl.col("player_id") == "DFL-OBJ-002G5S"
        ).sort("frame_id")

        # Player should appear starting at frame 10026 (27th frame, 0-indexed is frame 26)
        assert player_frames.height == 67
        # The 27th frame in period 1 would be frame 10026
        assert player_frames["frame_id"][0] == 10026


class TestTimestampBehavior:
    """Tests for timestamp and FPS behavior."""

    @pytest.fixture
    def tracking_df(self):
        """Load tracking data."""
        dataset = sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        tracking_df = dataset.tracking
        return tracking_df

    @pytest.fixture
    def metadata_df(self):
        """Load metadata."""
        dataset = sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        metadata_df = dataset.metadata
        return metadata_df

    def test_period_1_first_frame_timestamp_near_zero(self, tracking_df):
        """Test that period 1 first frame timestamp is at or near 0ms."""
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        if len(period_1) > 0:
            first_timestamp = period_1["timestamp"][0]
            # Should be at 0ms or very close (within first 100ms)
            # NOT wall-clock time (20+ hours)
            assert first_timestamp.total_seconds() * 1000 < 100

    def test_period_2_first_frame_timestamp_near_zero(self, tracking_df):
        """Test that period 2 first frame timestamp is at or near 0ms (period-relative)."""
        period_2 = tracking_df.filter(pl.col("period_id") == 2).sort("frame_id")
        if len(period_2) > 0:
            first_timestamp = period_2["timestamp"][0]
            # Should be at 0ms or very close (within first 100ms)
            # NOT wall-clock time (21+ hours)
            assert first_timestamp.total_seconds() * 1000 < 100

    def test_timestamp_matches_fps(self, tracking_df, metadata_df):
        """Test that timestamp increments match the FPS (25fps = 40ms per frame)."""
        fps = metadata_df["fps"][0]
        expected_delta_ms = 1000 / fps  # 40ms for 25fps

        # Get first few frames of period 1 with unique frame_ids
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        unique_frames = period_1.unique(subset=["frame_id"], maintain_order=True).head(5)

        if len(unique_frames) >= 2:
            ts1 = unique_frames["timestamp"][0].total_seconds() * 1000
            ts2 = unique_frames["timestamp"][1].total_seconds() * 1000
            delta = ts2 - ts1
            # Allow some tolerance (within 5ms)
            assert abs(delta - expected_delta_ms) < 5

    def test_metadata_fps_value(self, metadata_df):
        """Test that FPS is correctly reported as 25."""
        assert metadata_df["fps"][0] == pytest.approx(25.0, rel=0.01)


class TestBomHandling:
    """Tests for UTF-8 BOM handling in Sportec XML files."""

    def test_bom_metadata_loads(self):
        """Test that BOM-prefixed metadata XML loads without error."""
        dataset = sportec.load_tracking(RAW_DATA_PATH, META_DATA_BOM_PATH, lazy=False)
        assert dataset.tracking.height > 0

    def test_bom_tracking_loads(self):
        """Test that BOM-prefixed tracking XML loads without error."""
        dataset = sportec.load_tracking(RAW_DATA_BOM_PATH, META_DATA_PATH, lazy=False)
        assert dataset.tracking.height > 0

    def test_bom_matches_non_bom(self):
        """Test that BOM-prefixed files produce same results as non-BOM."""
        dataset_normal = sportec.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False, only_alive=False)
        dataset_bom = sportec.load_tracking(RAW_DATA_BOM_PATH, META_DATA_BOM_PATH, lazy=False, only_alive=False)

        assert dataset_bom.tracking.height == dataset_normal.tracking.height
        assert dataset_bom.players.height == dataset_normal.players.height
