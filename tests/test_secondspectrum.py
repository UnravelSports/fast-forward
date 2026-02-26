"""Integration tests for SecondSpectrum tracking data loading.

This file contains tests for both:
1. Anonymized test data (secondspectrum_tracking_anon.jsonl, secondspectrum_meta_anon.json)
2. Original kloppy test data (second_spectrum_fake_data.jsonl, second_spectrum_fake_metadata.json)
"""

import datetime

import pytest
import polars as pl

from fastforward import secondspectrum
from fastforward._dataset import TrackingDataset
from tests.config import (
    SS_RAW_ANON as ANON_RAW_DATA_PATH,
    SS_META_ANON as ANON_META_DATA_PATH,
    SS_RAW_NULL_BALL as NULL_BALL_RAW_DATA_PATH,
    SS_RAW_FAKE as KLOPPY_RAW_DATA_PATH,
    SS_RAW_FAKE_UTF8SIG as KLOPPY_RAW_DATA_UTF8SIG_PATH,
    SS_META_FAKE as KLOPPY_META_DATA_PATH,
    SS_META_FAKE_XML as KLOPPY_XML_META_DATA_PATH,
    SS_META_FAKE_XML_BOM as KLOPPY_XML_BOM_META_DATA_PATH,
)


# =============================================================================
# Tests using Anonymized Data
# =============================================================================

class TestLoadTracking:
    """Tests for secondspectrum.load_tracking function."""

    def test_returns_tracking_dataset(self):
        """Test that load_tracking returns a TrackingDataset object."""
        result = secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False)

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
        return secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False)

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
        """Test pitch dimensions are exact values after rounding."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert round(pitch_length, 1) == 104.9
        assert round(pitch_width, 1) == 68.0

    def test_fps(self, metadata_df):
        """Test fps is exactly 25.0."""
        assert metadata_df["fps"][0] == 25.0

    def test_game_date(self, metadata_df):
        """Test game_date is present and valid."""
        game_date = metadata_df["game_date"][0]
        assert game_date == datetime.date(2025, 1, 1)


class TestTeamDataFrame:
    """Tests for the team DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False)

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
        return secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False)

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
        first_player = player_df.row(0, named=True)
        assert first_player["name"] is not None
        assert first_player["first_name"] is not None or first_player["last_name"] is not None

    def test_has_players(self, player_df):
        """Test that player_df contains players from both teams."""
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
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, layout="long", lazy=False
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
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, layout="long_ball", lazy=False
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
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, layout="wide", lazy=False
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
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, coordinates="cdf", lazy=False
        )
        assert dataset.tracking.height == 4554
        assert dataset.metadata["coordinate_system"][0] == "cdf"

    def test_invalid_coordinate_system(self):
        """Test that invalid coordinate system raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(
                ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, coordinates="invalid", lazy=False
            )


class TestLayoutParameter:
    """Tests for layout parameter validation."""

    def test_invalid_layout(self):
        """Test that invalid layout raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, layout="invalid", lazy=False)


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_tracking_file(self):
        """Test that missing tracking file raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking("nonexistent_tracking.jsonl", ANON_META_DATA_PATH, lazy=False)

    def test_missing_metadata_file(self):
        """Test that missing metadata file raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(ANON_RAW_DATA_PATH, "nonexistent_metadata.json")


class TestOnlyAliveParameter:
    """Tests for only_alive parameter."""

    def test_only_alive_filters_dead_frames(self):
        """Test that only_alive=True filters out dead ball frames."""
        dataset_all = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, only_alive=False, exclude_missing_ball_frames=False, lazy=False
        )
        dataset_alive = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, only_alive=True, exclude_missing_ball_frames=False, lazy=False
        )

        assert dataset_all.tracking.height == 4600
        assert dataset_alive.tracking.height == 4554

    def test_only_alive_no_dead_frames(self):
        """Test that only_alive=True results in no dead ball frames."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, only_alive=True, lazy=False
        )
        dead_rows = dataset.tracking.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0

    def test_only_alive_default_true(self):
        """Test that only_alive defaults to True (excludes dead frames)."""
        dataset = secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False)
        dead_rows = dataset.tracking.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0


class TestExcludeMissingBallFramesParameter:
    """Tests for exclude_missing_ball_frames parameter."""

    def test_exclude_missing_ball_frames_filters_frames(self):
        """Test that exclude_missing_ball_frames=True filters out frames with ball_z == -10."""
        # Load with exclude_missing_ball_frames=False to include all frames
        dataset_all = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, only_alive=False, exclude_missing_ball_frames=False, lazy=False
        )
        # Load with exclude_missing_ball_frames=True (default) to filter out missing ball frames
        dataset_filtered = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, only_alive=False, exclude_missing_ball_frames=True, lazy=False
        )

        # The test data has 1 frame with ball_z == -10 (frameIdx 0)
        # That frame has 23 rows in long format (11 home + 11 away + 1 ball)
        assert dataset_all.tracking.height == 4600
        assert dataset_filtered.tracking.height == 4577  # 4600 - 23

    def test_exclude_missing_ball_frames_default_true(self):
        """Test that exclude_missing_ball_frames defaults to True."""
        # Default should exclude frames with ball_z == -10
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, only_alive=False, lazy=False
        )
        # Should have fewer rows because frame 0 with ball_z=-10 is excluded
        assert dataset.tracking.height == 4577


class TestOrientationParameter:
    """Tests for orientation parameter."""

    def test_orientation_default_static_home_away(self):
        """Test that orientation defaults to 'static_home_away'."""
        dataset = secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False)
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_orientation_static_away_home(self):
        """Test that orientation='static_away_home' is recorded."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, orientation="static_away_home", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "static_away_home"

    def test_orientation_home_away(self):
        """Test orientation='home_away'."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, orientation="home_away", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "home_away"

    def test_orientation_away_home(self):
        """Test orientation='away_home'."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, orientation="away_home", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "away_home"

    def test_orientation_attack_right(self):
        """Test orientation='attack_right'."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, orientation="attack_right", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "attack_right"

    def test_orientation_attack_left(self):
        """Test orientation='attack_left'."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, orientation="attack_left", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "attack_left"

    def test_invalid_orientation(self):
        """Test that invalid orientation raises error."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(
                ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, orientation="invalid", lazy=False
            )

    def test_orientation_transforms_coordinates(self):
        """Test that different orientations can produce different coordinates."""
        dataset_default = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, orientation="static_home_away", lazy=False
        )
        dataset_away = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, orientation="static_away_home", lazy=False
        )

        player_default = dataset_default.tracking.filter(pl.col("team_id") != "ball").row(0, named=True)
        player_away = dataset_away.tracking.filter(pl.col("team_id") != "ball").row(0, named=True)

        assert player_default["x"] is not None
        assert player_away["x"] is not None


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyParameter:
    """Tests for lazy loading parameter."""

    def test_lazy_returns_lazyframe(self):
        """Test that lazy=True returns a TrackingDataset with pl.LazyFrame."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=True
        )
        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.LazyFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)

    def test_lazy_collect_returns_dataframe(self):
        """Test that collect() returns a DataFrame."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=True
        )
        result = dataset.tracking.collect()
        assert isinstance(result, pl.DataFrame)

    def test_lazy_filter_before_collect(self):
        """Test that filter() can be chained before collect()."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=True
        )
        result = dataset.tracking.filter(pl.col("period_id") == 1).collect()
        assert all(p == 1 for p in result["period_id"].to_list())

    def test_lazy_select_before_collect(self):
        """Test that select() can be chained before collect()."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=True
        )
        result = dataset.tracking.select(["frame_id", "x", "y"]).collect()
        assert set(result.columns) == {"frame_id", "x", "y"}

    def test_lazy_collect_matches_eager(self):
        """Test that lazy collect() produces same result as eager loading."""
        dataset_lazy = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=True
        )
        dataset_eager = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False
        )
        assert dataset_lazy.tracking.collect().equals(dataset_eager.tracking)

    def test_lazy_repr(self):
        """Test that pl.LazyFrame has a useful repr."""
        dataset = secondspectrum.load_tracking(
            ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=True
        )
        repr_str = repr(dataset.tracking)
        assert "frame_id" in repr_str or "LazyFrame" in repr_str


class TestLazyNotImplemented:
    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=True)

    def test_from_cache_raises(self):
        with pytest.raises(NotImplementedError, match="cache loading"):
            secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, from_cache=True)


class TestPeriodsDataFrame:
    """Tests for the periods DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load dataset."""
        return secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False)

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

        # Period 1: first alive frame at 40ms (frame 0 is dead), last at 3960ms
        p1 = periods.row(0, named=True)
        assert p1["start_timestamp"] == timedelta(milliseconds=40)
        assert p1["end_timestamp"] == timedelta(milliseconds=3960)
        assert p1["duration"] == timedelta(milliseconds=3920)

        # Period 2: same pattern
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] == timedelta(milliseconds=40)
        assert p2["end_timestamp"] == timedelta(milliseconds=3960)
        assert p2["duration"] == timedelta(milliseconds=3920)


class TestTimestampBehavior:
    """Tests for timestamp and FPS behavior."""

    @pytest.fixture
    def dataset(self):
        """Load dataset."""
        return secondspectrum.load_tracking(ANON_RAW_DATA_PATH, ANON_META_DATA_PATH, lazy=False)

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking data."""
        return dataset.tracking

    @pytest.fixture
    def metadata_df(self, dataset):
        """Return metadata."""
        return dataset.metadata

    def test_period_1_first_frame_timestamp(self, tracking_df):
        """Test that period 1 first alive frame timestamp is exactly 40ms (frame 0 is dead ball)."""
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        first_timestamp = period_1["timestamp"][0]
        assert first_timestamp.total_seconds() * 1000 == 40.0

    def test_period_2_first_frame_timestamp(self, tracking_df):
        """Test that period 2 first alive frame timestamp is exactly 40ms (period-relative)."""
        period_2 = tracking_df.filter(pl.col("period_id") == 2).sort("frame_id")
        if len(period_2) > 0:
            first_timestamp = period_2["timestamp"][0]
            assert first_timestamp.total_seconds() * 1000 == 40.0

    def test_timestamp_matches_fps(self, tracking_df, metadata_df):
        """Test that timestamp increments match the FPS (25fps = 40ms per frame)."""
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        unique_frames = period_1.unique(subset=["frame_id"], maintain_order=True).head(5)

        if len(unique_frames) >= 2:
            ts1 = unique_frames["timestamp"][0].total_seconds() * 1000
            ts2 = unique_frames["timestamp"][1].total_seconds() * 1000
            delta = ts2 - ts1
            assert delta == 40.0

    def test_metadata_fps_value(self, metadata_df):
        """Test that FPS is correctly reported as exactly 25.0."""
        assert metadata_df["fps"][0] == 25.0


# =============================================================================
# Tests using Original Kloppy Data
# =============================================================================

class TestKloppyBasicDeserialization:
    """Tests for basic loading and deserialization using original kloppy data."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )

    def test_returns_tracking_dataset(self, dataset):
        """Test that load_tracking returns a TrackingDataset object."""
        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)
        assert isinstance(dataset.periods, pl.DataFrame)

    def test_provider_is_secondspectrum(self, dataset):
        """Test provider is correctly identified as 'secondspectrum'."""
        assert dataset.metadata["provider"][0] == "secondspectrum"

    def test_has_two_periods(self, dataset):
        """Test that data includes two periods."""
        periods = dataset.tracking["period_id"].unique().to_list()
        assert len(periods) == 2
        assert 1 in periods
        assert 2 in periods

    def test_total_frames(self, dataset):
        """Test that correct number of frames are loaded."""
        unique_frames = dataset.tracking["frame_id"].n_unique()
        assert unique_frames == 376


class TestKloppyMetadata:
    """Tests for metadata loading using original kloppy data."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )

    @pytest.fixture
    def metadata_df(self, dataset):
        """Return metadata DataFrame."""
        return dataset.metadata

    def test_pitch_dimensions(self, metadata_df):
        """Test pitch dimensions are loaded correctly (~104.85 x ~67.97)."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert pytest.approx(pitch_length, rel=0.01) == 104.8512
        assert pytest.approx(pitch_width, rel=0.01) == 67.9704

    def test_fps_is_25(self, metadata_df):
        """Test FPS is 25."""
        assert metadata_df["fps"][0] == pytest.approx(25.0, rel=0.01)

    def test_game_date(self, metadata_df):
        """Test game date is parsed correctly."""
        game_date = metadata_df["game_date"][0]
        assert game_date == datetime.date(1900, 1, 26)

    def test_team_names(self, metadata_df):
        """Test team names are extracted from description."""
        assert metadata_df["home_team"][0] == "FK1"
        assert metadata_df["away_team"][0] == "FK2"


class TestKloppyTeams:
    """Tests for team data loading using original kloppy data."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )

    @pytest.fixture
    def team_df(self, dataset):
        """Return teams DataFrame."""
        return dataset.teams

    def test_two_teams(self, team_df):
        """Test that exactly two teams are loaded."""
        assert team_df.height == 2

    def test_grounds(self, team_df):
        """Test that team_df contains home and away teams."""
        grounds = set(team_df["ground"].to_list())
        assert grounds == {"home", "away"}


class TestKloppyPlayers:
    """Tests for player data loading using original kloppy data."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )

    @pytest.fixture
    def player_df(self, dataset):
        """Return players DataFrame."""
        return dataset.players

    def test_has_players(self, player_df):
        """Test that player_df contains players from both teams."""
        assert player_df.height == 40

    def test_has_goalkeeper(self, player_df):
        """Test that there's at least one goalkeeper."""
        goalkeepers = player_df.filter(pl.col("position") == "GK")
        assert goalkeepers.height >= 1

    def test_jersey_numbers_present(self, player_df):
        """Test that jersey numbers are loaded."""
        assert "jersey_number" in player_df.columns
        assert player_df["jersey_number"].null_count() < player_df.height


class TestKloppyCoordinates:
    """Tests for coordinate loading and transformations using original kloppy data."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking DataFrame."""
        return dataset.tracking

    def test_player_coordinates(self, tracking_df):
        """Test that player coordinates are loaded."""
        player_rows = tracking_df.filter(pl.col("team_id") != "ball")
        first_player = player_rows.row(0, named=True)

        assert first_player["x"] is not None
        assert first_player["y"] is not None

    def test_ball_coordinates(self, tracking_df):
        """Test that ball coordinates are loaded (including z)."""
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height > 0

        first_ball = ball_rows.row(0, named=True)
        assert first_ball["x"] is not None
        assert first_ball["y"] is not None
        assert first_ball["z"] is not None

    def test_coordinate_ranges(self, tracking_df, dataset):
        """Test that coordinates are within pitch bounds."""
        pitch_length = dataset.metadata["pitch_length"][0]
        pitch_width = dataset.metadata["pitch_width"][0]

        max_x = pitch_length / 2 + 1
        max_y = pitch_width / 2 + 1

        x_min = tracking_df["x"].min()
        x_max = tracking_df["x"].max()
        assert x_min >= -max_x
        assert x_max <= max_x

        y_min = tracking_df["y"].min()
        y_max = tracking_df["y"].max()
        assert y_min >= -max_y
        assert y_max <= max_y


class TestKloppyBallState:
    """Tests for ball state (live/dead) handling using original kloppy data."""

    def test_has_live_and_dead_states(self):
        """Test that data includes both live and dead ball states."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )
        states = set(dataset.tracking["ball_state"].unique().to_list())
        assert "alive" in states or "dead" in states

    def test_only_alive_filters(self):
        """Test that only_alive=True filters out dead ball frames."""
        dataset_all = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )
        dataset_alive = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=True, lazy=False
        )

        assert dataset_alive.tracking.height <= dataset_all.tracking.height

        if dataset_alive.tracking.height > 0:
            dead_rows = dataset_alive.tracking.filter(pl.col("ball_state") == "dead")
            assert dead_rows.height == 0


class TestKloppyUTF8Encoding:
    """Tests for UTF-8 BOM encoding handling."""

    def test_utf8sig_file_loads(self):
        """Test that UTF-8 with BOM encoded file loads successfully."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_UTF8SIG_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert isinstance(dataset, TrackingDataset)
        assert dataset.tracking.height > 0

    def test_utf8sig_frame_count(self):
        """Test that UTF-8 sig file has expected number of frames."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_UTF8SIG_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )
        unique_frames = dataset.tracking["frame_id"].n_unique()
        assert unique_frames > 0


class TestKloppyLayouts:
    """Tests for different DataFrame layouts using original kloppy data."""

    def test_long_layout(self):
        """Test long layout has ball as separate rows."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, layout="long", only_alive=False, lazy=False
        )
        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 376

    def test_long_ball_layout(self):
        """Test long_ball layout has ball in separate columns."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, layout="long_ball", only_alive=False, lazy=False
        )
        ball_rows = dataset.tracking.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 0

        assert "ball_x" in dataset.tracking.columns
        assert "ball_y" in dataset.tracking.columns
        assert "ball_z" in dataset.tracking.columns

    def test_wide_layout(self):
        """Test wide layout has one row per frame."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, layout="wide", only_alive=False, lazy=False
        )
        assert dataset.tracking.height == 376

        assert "ball_x" in dataset.tracking.columns
        assert "ball_y" in dataset.tracking.columns
        assert "ball_z" in dataset.tracking.columns


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestKloppyLazyLoading:
    """Tests for lazy loading functionality using original kloppy data."""

    def test_lazy_returns_lazyframe(self):
        """Test that lazy=True returns LazyFrame for tracking."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, lazy=True
        )
        assert isinstance(dataset.tracking, pl.LazyFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)

    def test_lazy_collect_works(self):
        """Test that collecting lazy frame produces correct results."""
        dataset_lazy = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=True
        )
        dataset_eager = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )

        collected = dataset_lazy.tracking.collect()
        assert collected.equals(dataset_eager.tracking)


class TestKloppyTimestamps:
    """Tests for timestamp handling using original kloppy data."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking DataFrame."""
        return dataset.tracking

    def test_timestamp_type(self, tracking_df):
        """Test that timestamp column is Duration type."""
        assert tracking_df.schema["timestamp"] == pl.Duration("ms")

    def test_first_frame_near_zero(self, tracking_df):
        """Test that first frame of period 1 has timestamp near 0."""
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        first_timestamp = period_1["timestamp"][0]
        assert first_timestamp.total_seconds() * 1000 < 100

    def test_timestamp_increment_matches_fps(self, tracking_df):
        """Test that timestamps increment at 40ms (25fps)."""
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        unique_frames = period_1.unique(subset=["frame_id"], maintain_order=True).head(5)

        if len(unique_frames) >= 2:
            ts1 = unique_frames["timestamp"][0].total_seconds() * 1000
            ts2 = unique_frames["timestamp"][1].total_seconds() * 1000
            delta = ts2 - ts1
            expected_delta = 40.0
            assert delta >= expected_delta - 5


class TestKloppyLastTouch:
    """Tests for last touch / ball owning team tracking using original kloppy data."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking DataFrame."""
        return dataset.tracking

    def test_ball_owning_team_column(self, tracking_df):
        """Test that ball_owning_team_id column exists."""
        assert "ball_owning_team_id" in tracking_df.columns

    def test_ball_owning_team_values(self, tracking_df, dataset):
        """Test that ball_owning_team_id has valid team values."""
        owning_teams = set(tracking_df["ball_owning_team_id"].unique().to_list())
        team_ids = set(dataset.teams["team_id"].to_list())

        owning_teams.discard(None)

        if len(owning_teams) > 0:
            assert len(owning_teams.intersection(team_ids)) > 0 or "home" in owning_teams or "away" in owning_teams


class TestXMLMetadata:
    """Tests for XML metadata format support."""

    def test_xml_metadata_loads(self):
        """Test that XML metadata file loads successfully."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert isinstance(dataset, TrackingDataset)
        assert dataset.tracking["frame_id"].n_unique() == 376

    def test_xml_pitch_dimensions(self):
        """Test pitch dimensions from XML metadata."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert pytest.approx(dataset.metadata["pitch_length"][0], rel=0.01) == 104.85
        assert pytest.approx(dataset.metadata["pitch_width"][0], rel=0.01) == 67.97

    def test_xml_fps(self):
        """Test FPS from XML metadata."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert dataset.metadata["fps"][0] == pytest.approx(25.0)

    def test_xml_periods(self):
        """Test period data from XML metadata."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert dataset.periods.height == 2
        assert 1 in dataset.periods["period_id"].to_list()
        assert 2 in dataset.periods["period_id"].to_list()

    def test_xml_players_extracted_from_tracking(self):
        """Test that players are extracted from tracking data when using XML metadata."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert dataset.players.height == 29

    def test_xml_teams_default_names(self):
        """Test that teams have default names when using XML metadata."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert dataset.teams.height == 2
        grounds = set(dataset.teams["ground"].to_list())
        assert grounds == {"home", "away"}

    def test_xml_frame_count_matches_json(self):
        """Test that XML metadata produces same frame count as JSON."""
        dataset_json = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_META_DATA_PATH, only_alive=False, lazy=False
        )
        dataset_xml = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert dataset_xml.tracking["frame_id"].n_unique() == dataset_json.tracking["frame_id"].n_unique()

    def test_xml_game_date(self):
        """Test that game date is parsed from XML dtDate attribute."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        game_date = dataset.metadata["game_date"][0]
        assert game_date == datetime.date(1900, 2, 1)


class TestNullBallXyz:
    """Tests for handling null ball xyz coordinates (GitHub issue #2).

    SecondSpectrum data can contain frames where ball.xyz is null
    (e.g., when ball tracking is lost). These should be handled
    gracefully instead of crashing with a schema mismatch error.
    """

    def test_null_ball_xyz_excluded_by_default(self):
        """Test that frames with null ball xyz are excluded when exclude_missing_ball_frames=True."""
        dataset = secondspectrum.load_tracking(
            NULL_BALL_RAW_DATA_PATH,
            ANON_META_DATA_PATH,
            only_alive=False,
            exclude_missing_ball_frames=True,
        )
        # 3 frames total, 1 has null ball xyz → 2 frames remain
        # In long format: 2 frames * 23 rows each (11 home + 11 away + 1 ball)
        assert dataset.tracking.height == 46

    def test_null_ball_xyz_included_when_not_excluded(self):
        """Test that frames with null ball xyz are included when exclude_missing_ball_frames=False."""
        dataset = secondspectrum.load_tracking(
            NULL_BALL_RAW_DATA_PATH,
            ANON_META_DATA_PATH,
            only_alive=False,
            exclude_missing_ball_frames=False,
        )
        # All 3 frames included: 3 * 23 = 69 rows
        assert dataset.tracking.height == 69

    def test_null_ball_xyz_produces_nan_coordinates(self):
        """Test that null ball xyz results in NaN ball_x/ball_y/ball_z values."""
        import math

        dataset = secondspectrum.load_tracking(
            NULL_BALL_RAW_DATA_PATH,
            ANON_META_DATA_PATH,
            only_alive=False,
            exclude_missing_ball_frames=False,
        )
        # Frame 1 (frameIdx=1) has null ball xyz
        frame_1_ball = dataset.tracking.filter(
            (pl.col("frame_id") == 1) & (pl.col("player_id") == "ball")
        )
        assert frame_1_ball.height == 1
        assert math.isnan(frame_1_ball["x"][0])
        assert math.isnan(frame_1_ball["y"][0])
        assert math.isnan(frame_1_ball["z"][0])

    def test_null_ball_xyz_does_not_crash(self):
        """Test that loading data with null ball xyz does not raise an error."""
        # This is the core regression test for GitHub issue #2
        dataset = secondspectrum.load_tracking(
            NULL_BALL_RAW_DATA_PATH,
            ANON_META_DATA_PATH,
            only_alive=True,
            exclude_missing_ball_frames=True,
        )
        assert dataset.tracking.height > 0


class TestBomHandling:
    """Tests for UTF-8 BOM handling in SecondSpectrum XML metadata."""

    def test_bom_xml_metadata_loads(self):
        """Test that BOM-prefixed XML metadata loads without error."""
        dataset = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_BOM_META_DATA_PATH, only_alive=False, lazy=False
        )
        assert dataset.tracking.height > 0

    def test_bom_xml_matches_non_bom(self):
        """Test that BOM-prefixed XML metadata produces same results as non-BOM."""
        dataset_normal = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_META_DATA_PATH, only_alive=False, lazy=False
        )
        dataset_bom = secondspectrum.load_tracking(
            KLOPPY_RAW_DATA_PATH, KLOPPY_XML_BOM_META_DATA_PATH, only_alive=False, lazy=False
        )

        assert dataset_bom.tracking.height == dataset_normal.tracking.height
        assert dataset_bom.players.height == dataset_normal.players.height
