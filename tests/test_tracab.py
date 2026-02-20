"""Integration tests for Tracab tracking data loading."""

import pytest
import polars as pl
import datetime

from fastforward import tracab
from tests.config import (
    TR_META_XML as META_XML_PATH,
    TR_META_JSON as META_JSON_PATH,
    TR_META_XML_2 as META_XML_2_PATH,
    TR_META_XML_3 as META_XML_3_PATH,
    TR_META_XML_4 as META_XML_4_PATH,
    TR_RAW_DAT as RAW_DAT_PATH,
    TR_RAW_JSON as RAW_JSON_PATH,
)


class TestLoadTracking:
    """Tests for tracab.load_tracking function."""

    def test_returns_dataset(self):
        """Test that load_tracking returns a TrackingDataset."""
        from fastforward import TrackingDataset

        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)

        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)
        assert isinstance(dataset.periods, pl.DataFrame)

    def test_dat_with_xml_metadata(self):
        """Test DAT raw data with XML metadata."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)
        assert dataset.tracking.height == 46

    def test_dat_with_json_metadata(self):
        """Test DAT raw data with JSON metadata."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_JSON_PATH, lazy=False)
        assert dataset.tracking.height == 46

    def test_json_with_xml_metadata(self):
        """Test JSON raw data with XML metadata."""
        dataset = tracab.load_tracking(RAW_JSON_PATH, META_XML_PATH, lazy=False)
        assert dataset.tracking.height == 46

    def test_json_with_json_metadata(self):
        """Test JSON raw data with JSON metadata."""
        dataset = tracab.load_tracking(RAW_JSON_PATH, META_JSON_PATH, lazy=False)
        assert dataset.tracking.height == 46


class TestMetadataDataFrame:
    """Tests for the metadata DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)

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
        """Test provider is 'tracab'."""
        assert metadata_df["provider"][0] == "tracab"

    def test_coordinate_system_value(self, metadata_df):
        """Test coordinate_system is 'cdf'."""
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_orientation_value(self, metadata_df):
        """Test orientation is 'static_home_away'."""
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_team_names(self, metadata_df):
        """Test team names are extracted correctly."""
        assert metadata_df["home_team"][0] == "Short Name Home"
        assert metadata_df["away_team"][0] == "Short Name Away"

    def test_pitch_dimensions(self, metadata_df):
        """Test pitch dimensions are correct."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert pitch_length == pytest.approx(105.0, rel=0.01)
        assert pitch_width == pytest.approx(68.0, rel=0.01)

    def test_fps(self, metadata_df):
        """Test fps is 25."""
        assert metadata_df["fps"][0] == pytest.approx(25.0, rel=0.01)

    def test_game_id(self, metadata_df):
        """Test game_id is extracted correctly."""
        assert metadata_df["game_id"][0] == "1"

    def test_game_date(self, metadata_df):
        """Test game_date is present and valid."""
        game_date = metadata_df["game_date"][0]
        assert game_date == datetime.date(2023, 12, 15)


class TestTeamDataFrame:
    """Tests for the team DataFrame."""

    @pytest.fixture
    def team_df(self):
        """Load and return the team DataFrame."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)
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
    def player_df(self):
        """Load and return the player DataFrame."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)
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
        """Test that we have players from both teams."""
        # Should have 20 players per team = 40 total
        assert player_df.height == 40


class TestPeriodsDataFrame:
    """Tests for the periods DataFrame."""

    @pytest.fixture
    def periods_df(self):
        """Load and return the periods DataFrame."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)
        return dataset.periods

    def test_two_periods(self, periods_df):
        """Test that we have 2 periods."""
        assert periods_df.height == 2

    def test_period_ids(self, periods_df):
        """Test period IDs are correct."""
        period_ids = periods_df["period_id"].to_list()
        assert period_ids == [1, 2]

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

    def test_period_timing(self, periods_df):
        """Test that all periods have correct timing values."""
        from datetime import timedelta

        periods = periods_df.sort("period_id")

        # Period 1: test data has 2 frames at 40ms and 80ms
        p1 = periods.row(0, named=True)
        assert p1["start_timestamp"] == timedelta(milliseconds=40)
        assert p1["end_timestamp"] == timedelta(milliseconds=80)
        assert p1["duration"] == timedelta(milliseconds=40)

        # Period 2: no tracking frames in test data
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] is None
        assert p2["end_timestamp"] is None
        assert p2["duration"] is None


class TestTrackingDataFrameLong:
    """Tests for the tracking DataFrame in long layout."""

    @pytest.fixture
    def tracking_df(self):
        """Load and return the tracking DataFrame."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False, layout="long")
        return dataset.tracking

    def test_schema(self, tracking_df):
        """Test that tracking_df has expected columns."""
        expected_columns = {
            "game_id",
            "period_id",
            "timestamp",
            "frame_id",
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
        """Test that ball data is in rows with team_id='ball'."""
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 2

    def test_has_player_rows(self, tracking_df):
        """Test that player rows exist."""
        player_rows = tracking_df.filter(pl.col("team_id") != "ball")
        assert player_rows.height == 44


class TestTrackingDataFrameLongBall:
    """Tests for the tracking DataFrame in long_ball layout."""

    @pytest.fixture
    def tracking_df(self):
        """Load and return the tracking DataFrame."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False, layout="long_ball")
        return dataset.tracking

    def test_ball_columns(self, tracking_df):
        """Test that ball data is in separate columns."""
        assert "ball_x" in tracking_df.columns
        assert "ball_y" in tracking_df.columns
        assert "ball_z" in tracking_df.columns

    def test_no_ball_team_id(self, tracking_df):
        """Test that there are no ball rows (team_id='ball')."""
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 0


class TestTrackingDataFrameWide:
    """Tests for the tracking DataFrame in wide layout."""

    @pytest.fixture
    def tracking_df(self):
        """Load and return the tracking DataFrame."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False, layout="wide")
        return dataset.tracking

    def test_one_row_per_frame(self, tracking_df):
        """Test that there's one row per frame (not per player)."""
        # Wide layout should have fewer rows than long
        long_dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False, layout="long")
        assert tracking_df.height < long_dataset.tracking.height


class TestOnlyAliveParameter:
    """Tests for the only_alive parameter."""

    def test_only_alive_true_filters_dead(self):
        """Test that only_alive=True filters dead frames."""
        alive_dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, only_alive=True
        )
        not_alive_dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, only_alive=False
        )

        # With only_alive=True should have fewer or equal rows
        alive_frames = alive_dataset.tracking.select("frame_id").unique()
        all_frames = not_alive_dataset.tracking.select("frame_id").unique()
        assert alive_frames.height == 2
        assert all_frames.height == 7

    def test_only_alive_true_no_dead_state(self):
        """Test that only_alive=True removes dead ball states."""
        dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, only_alive=True
        )
        ball_states = dataset.tracking.filter(pl.col("team_id") == "ball")["ball_state"].unique().to_list()
        assert "dead" not in ball_states


class TestCoordinateParameter:
    """Tests for the coordinates parameter."""

    def test_cdf_coordinates(self):
        """Test CDF coordinate values (meters, origin at center)."""
        dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, coordinates="cdf"
        )
        tracking_df = dataset.tracking

        # CDF uses meters, values should be in reasonable range (-52.5 to 52.5 for x)
        x_values = tracking_df["x"].drop_nulls()
        if len(x_values) > 0:
            # CDF values are in meters, max around 52.5
            assert x_values.max() < 60
            assert x_values.min() > -60

    def test_tracab_coordinates(self):
        """Test Tracab coordinate values (centimeters, origin at center)."""
        dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, coordinates="tracab"
        )
        tracking_df = dataset.tracking

        # Tracab uses centimeters, values should be in range (-5250 to 5250 for x)
        x_values = tracking_df["x"].drop_nulls()
        # Tracab values are in centimeters (100x larger than CDF)
        assert x_values.max() == pytest.approx(4722.0, rel=0.01)
        assert x_values.min() == pytest.approx(-5270.0, rel=0.01)


class TestOrientationParameter:
    """Tests for the orientation parameter."""

    def test_static_home_away(self):
        """Test static_home_away orientation."""
        dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, orientation="static_home_away"
        )
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_static_away_home(self):
        """Test static_away_home orientation."""
        dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, orientation="static_away_home"
        )
        assert dataset.metadata["orientation"][0] == "static_away_home"


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyParameter:
    """Tests for the lazy parameter."""

    def test_lazy_true_returns_lazyframe(self):
        """Test that lazy=True returns pl.LazyFrame for tracking."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=True)
        assert isinstance(dataset.tracking, pl.LazyFrame)

    def test_lazy_false_returns_dataframe(self):
        """Test that lazy=False returns DataFrame directly."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)
        assert isinstance(dataset.tracking, pl.DataFrame)

    def test_lazy_collect(self):
        """Test that lazy loader can be collected."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=True)
        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert tracking_df.height == 46


class TestLazyNotImplemented:
    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=True)

    def test_from_cache_raises(self):
        with pytest.raises(NotImplementedError, match="cache loading"):
            tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, from_cache=True)


class TestIncludeGameIdParameter:
    """Tests for the include_game_id parameter."""

    def test_include_game_id_true(self):
        """Test that include_game_id=True includes game_id column."""
        dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, include_game_id=True
        )
        assert "game_id" in dataset.tracking.columns
        assert dataset.tracking["game_id"][0] == "1"

    def test_include_game_id_false(self):
        """Test that include_game_id=False excludes game_id column."""
        dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, include_game_id=False
        )
        assert "game_id" not in dataset.tracking.columns

    def test_include_game_id_custom_string(self):
        """Test that include_game_id=str uses custom value."""
        dataset = tracab.load_tracking(
            RAW_DAT_PATH, META_XML_PATH, lazy=False, include_game_id="custom_id"
        )
        assert "game_id" in dataset.tracking.columns
        assert dataset.tracking["game_id"][0] == "custom_id"


class TestTimestampBehavior:
    """Tests for timestamp behavior (period-relative)."""

    def test_timestamps_reset_per_period(self):
        """Test that timestamps reset at start of each period."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False, only_alive=False)
        tracking_df = dataset.tracking

        for period_id in [1, 2]:
            period_data = tracking_df.filter(pl.col("period_id") == period_id)
            if period_data.height > 0:
                first_timestamp = period_data.sort("timestamp")["timestamp"][0]
                # First frame of each period should start at 0 (period-relative)
                # Check that it's a duration starting at 0
                assert first_timestamp.total_seconds() < 1  # Less than 1 second from period start


class TestMetadataFormats:
    """Tests for different metadata format variants."""

    def test_hierarchical_xml(self):
        """Test hierarchical XML format (tracab_meta.xml)."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)
        assert dataset.metadata["provider"][0] == "tracab"
        assert dataset.metadata["fps"][0] == pytest.approx(25.0)

    def test_minimal_xml(self):
        """Test minimal XML format (tracab_meta_2.xml) - no team/player data."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_2_PATH, lazy=False)
        assert dataset.metadata["provider"][0] == "tracab"

    def test_opta_format_xml(self):
        """Test Opta-style XML format (tracab_meta_3.xml)."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_3_PATH, lazy=False)
        assert dataset.metadata["provider"][0] == "tracab"

    def test_flat_xml(self):
        """Test flat XML format (tracab_meta_4.xml) with Phase elements."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_4_PATH, lazy=False)
        assert dataset.metadata["provider"][0] == "tracab"

    def test_json_metadata(self):
        """Test JSON metadata format."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_JSON_PATH, lazy=False)
        assert dataset.metadata["provider"][0] == "tracab"
        assert dataset.metadata["fps"][0] == pytest.approx(25.0)


class TestRawDataFormats:
    """Tests for different raw data formats."""

    def test_dat_format(self):
        """Test DAT raw data format."""
        dataset = tracab.load_tracking(RAW_DAT_PATH, META_XML_PATH, lazy=False)
        assert dataset.tracking.height == 46

    def test_json_format(self):
        """Test JSON raw data format."""
        dataset = tracab.load_tracking(RAW_JSON_PATH, META_XML_PATH, lazy=False)
        assert dataset.tracking.height == 46

    def test_formats_produce_similar_results(self):
        """Test that DAT and JSON produce similar frame counts."""
        dat_dataset = tracab.load_tracking(RAW_DAT_PATH, META_JSON_PATH, lazy=False, only_alive=False)
        json_dataset = tracab.load_tracking(RAW_JSON_PATH, META_JSON_PATH, lazy=False, only_alive=False)

        dat_frames = dat_dataset.tracking.select("frame_id").unique().height
        json_frames = json_dataset.tracking.select("frame_id").unique().height

        # Should have similar number of frames (may differ due to parsing differences)
        # Allow 10% tolerance
        assert abs(dat_frames - json_frames) / max(dat_frames, json_frames) < 0.1
