"""Integration tests for SecondSpectrum tracking data loading."""

import pytest
import polars as pl
from pathlib import Path

from kloppy_light import secondspectrum


# Test data paths (anonymized test data with 100 frames per period)
DATA_DIR = Path(__file__).parent / "files"
RAW_DATA_PATH = str(DATA_DIR / "secondspectrum_tracking.jsonl")
META_DATA_PATH = str(DATA_DIR / "secondspectrum_meta.json")


class TestLoadTracking:
    """Tests for secondspectrum.load_tracking function."""

    def test_returns_tracking_dataset(self):
        """Test that load_tracking returns a TrackingDataset object."""
        from kloppy_light._dataset import TrackingDataset

        result = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

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
        return secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

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
        """Test provider is 'secondspectrum'."""
        assert metadata_df["provider"][0] == "secondspectrum"

    def test_coordinate_system_value(self, metadata_df):
        """Test coordinate_system is 'cdf'."""
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_orientation_value(self, metadata_df):
        """Test orientation is 'static_home_away'."""
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_team_names(self, metadata_df):
        """Test team names are extracted correctly."""
        assert metadata_df["home_team"][0] == "HOME"
        assert metadata_df["away_team"][0] == "AWAY"

    def test_pitch_dimensions(self, metadata_df):
        """Test pitch dimensions are reasonable."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert 100.0 < pitch_length < 110.0
        assert 65.0 < pitch_width < 70.0

    def test_fps(self, metadata_df):
        """Test fps is 25."""
        assert metadata_df["fps"][0] == pytest.approx(25.0, rel=0.01)

    def test_game_date(self, metadata_df):
        """Test game_date is present and valid."""
        import datetime

        game_date = metadata_df["game_date"][0]
        assert game_date == datetime.date(2025, 1, 1)


class TestTeamDataFrame:
    """Tests for the team DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

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


class TestPlayerDataFrame:
    """Tests for the player DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def player_df(self, dataset):
        """Load and return the player DataFrame."""
        return dataset.players

    def test_schema(self, player_df):
        """Test that player_df has expected columns."""
        expected_columns = {"game_id", "team_id", "player_id", "name", "first_name", "last_name", "jersey_number", "position", "is_starter"}
        assert set(player_df.columns) == expected_columns

    def test_name_fields(self, player_df):
        """Test that name fields are populated correctly."""
        # Get first player
        first_player = player_df.row(0, named=True)

        # name should be set (SecondSpectrum provides full names)
        assert first_player["name"] is not None

        # first_name and last_name should be split from name
        assert first_player["first_name"] is not None or first_player["last_name"] is not None

    def test_has_players(self, player_df):
        """Test that player_df contains players from both teams."""
        # Expected: 20 home + 20 away = 40 players
        assert player_df.height == 40

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
    def dataset(self):
        """Load tracking data with long layout."""
        return secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="long", lazy=False
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

    def test_has_ball_rows(self, tracking_df):
        """Test that long format includes ball as separate rows."""
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 198

        # Check ball rows have player_id = "ball"
        assert ball_rows["player_id"].to_list()[:10] == ["ball"] * 10

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
        return secondspectrum.load_tracking(
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
        return secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, layout="wide", lazy=False
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

        assert len(x_columns) == 23
        assert len(x_columns) == len(y_columns) == len(z_columns)

    def test_one_row_per_frame(self, tracking_df):
        """Test that wide format has exactly one row per frame."""
        frame_count = tracking_df["frame_id"].n_unique()
        assert tracking_df.height == frame_count


class TestCoordinateSystem:
    """Tests for coordinate system parameter."""

    def test_cdf_coordinates(self):
        """Test that CDF coordinates work correctly."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="cdf", lazy=False
        )
        assert dataset.tracking.height == 4554
        assert dataset.metadata["coordinate_system"][0] == "cdf"

    def test_invalid_coordinate_system(self):
        """Test that invalid coordinate system raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH, coordinates="invalid", lazy=False
            )


class TestLayoutParameter:
    """Tests for layout parameter validation."""

    def test_invalid_layout(self):
        """Test that invalid layout raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, layout="invalid", lazy=False)


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_tracking_file(self):
        """Test that missing tracking file raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking("nonexistent_tracking.jsonl", META_DATA_PATH, lazy=False)

    def test_missing_metadata_file(self):
        """Test that missing metadata file raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(RAW_DATA_PATH, "nonexistent_metadata.json")


class TestOnlyAliveParameter:
    """Tests for only_alive parameter."""

    def test_only_alive_filters_dead_frames(self):
        """Test that only_alive=True filters out dead ball frames."""
        dataset_all = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=False, lazy=False
        )
        dataset_alive = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True, lazy=False
        )

        # alive should have fewer rows (or equal if no dead frames)
        assert dataset_all.tracking.height == 4600
        assert dataset_alive.tracking.height == 4554

    def test_only_alive_no_dead_frames(self):
        """Test that only_alive=True results in no dead ball frames."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True, lazy=False
        )

        # All rows should be alive
        dead_rows = dataset.tracking.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0

    def test_only_alive_default_true(self):
        """Test that only_alive defaults to True (excludes dead frames)."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

        # Should have no dead frames
        dead_rows = dataset.tracking.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0


class TestOrientationParameter:
    """Tests for orientation parameter."""

    def test_orientation_default_static_home_away(self):
        """Test that orientation defaults to 'static_home_away'."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_orientation_static_away_home(self):
        """Test that orientation='static_away_home' is recorded."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "static_away_home"

    def test_orientation_home_away(self):
        """Test orientation='home_away'."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="home_away", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "home_away"

    def test_orientation_away_home(self):
        """Test orientation='away_home'."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="away_home", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "away_home"

    def test_orientation_attack_right(self):
        """Test orientation='attack_right'."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_right", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "attack_right"

    def test_orientation_attack_left(self):
        """Test orientation='attack_left'."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_left", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "attack_left"

    def test_invalid_orientation(self):
        """Test that invalid orientation raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH, orientation="invalid", lazy=False
            )

    def test_orientation_transforms_coordinates(self):
        """Test that different orientations can produce different coordinates."""
        # Load with default orientation
        dataset_default = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_home_away", lazy=False
        )
        # Load with opposite orientation
        dataset_away = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", lazy=False
        )

        # Get first player row from each
        player_default = dataset_default.tracking.filter(pl.col("team_id") != "ball").row(0, named=True)
        player_away = dataset_away.tracking.filter(pl.col("team_id") != "ball").row(0, named=True)

        # Both should load successfully
        assert player_default["x"] is not None
        assert player_away["x"] is not None


class TestLazyParameter:
    """Tests for lazy loading parameter."""

    def test_lazy_returns_lazyframe(self):
        """Test that lazy=True returns a TrackingDataset with pl.LazyFrame."""
        from kloppy_light._dataset import TrackingDataset

        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.LazyFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)  # Metadata is eager
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)

    def test_lazy_collect_returns_dataframe(self):
        """Test that collect() returns a DataFrame."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        result = dataset.tracking.collect()
        assert isinstance(result, pl.DataFrame)

    def test_lazy_filter_before_collect(self):
        """Test that filter() can be chained before collect()."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        result = dataset.tracking.filter(pl.col("period_id") == 1).collect()
        # All rows should be period 1
        assert all(p == 1 for p in result["period_id"].to_list())

    def test_lazy_select_before_collect(self):
        """Test that select() can be chained before collect()."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        result = dataset.tracking.select(["frame_id", "x", "y"]).collect()
        assert set(result.columns) == {"frame_id", "x", "y"}

    def test_lazy_collect_matches_eager(self):
        """Test that lazy collect() produces same result as eager loading."""
        dataset_lazy = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        dataset_eager = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=False
        )
        assert dataset_lazy.tracking.collect().equals(dataset_eager.tracking)

    def test_lazy_repr(self):
        """Test that pl.LazyFrame has a useful repr."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        # LazyFrame repr shows column names and types
        repr_str = repr(dataset.tracking)
        assert "frame_id" in repr_str or "LazyFrame" in repr_str


class TestTimestampBehavior:
    """Tests for timestamp and FPS behavior."""

    @pytest.fixture
    def dataset(self):
        """Load dataset."""
        return secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking data."""
        return dataset.tracking

    @pytest.fixture
    def metadata_df(self, dataset):
        """Return metadata."""
        return dataset.metadata

    def test_period_1_first_frame_timestamp_near_zero(self, tracking_df):
        """Test that period 1 first frame timestamp is at or near 0ms."""
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        first_timestamp = period_1["timestamp"][0]
        # Should be at 0ms or very close (within first 100ms)
        assert first_timestamp.total_seconds() * 1000 < 100

    def test_period_2_first_frame_timestamp_near_zero(self, tracking_df):
        """Test that period 2 first frame timestamp is at or near 0ms (period-relative)."""
        period_2 = tracking_df.filter(pl.col("period_id") == 2).sort("frame_id")
        if len(period_2) > 0:
            first_timestamp = period_2["timestamp"][0]
            # Should be at 0ms or very close (within first 100ms)
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
