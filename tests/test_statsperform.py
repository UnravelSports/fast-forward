"""Integration tests for StatsPerform (Opta/SportVU) tracking data loading."""

import pytest
import polars as pl

from fastforward import statsperform
from tests.config import (
    STP_RAW_MA25 as MA25_DATA_PATH,
    STP_META_JSON as MA1_JSON_PATH,
    STP_META_XML as MA1_XML_PATH,
    STP_META_XML_BOM as MA1_XML_BOM_PATH,
)


class TestLoadTrackingWithJSON:
    """Tests for statsperform.load_tracking function with JSON metadata."""

    def test_returns_tracking_dataset(self):
        """Test that load_tracking returns a TrackingDataset object."""
        from fastforward._dataset import TrackingDataset

        result = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, lazy=False
        )

        assert isinstance(result, TrackingDataset)
        assert isinstance(result.tracking, pl.DataFrame)
        assert isinstance(result.metadata, pl.DataFrame)
        assert isinstance(result.teams, pl.DataFrame)
        assert isinstance(result.players, pl.DataFrame)
        assert isinstance(result.periods, pl.DataFrame)


class TestLoadTrackingWithXML:
    """Tests for statsperform.load_tracking function with XML metadata."""

    def test_returns_tracking_dataset(self):
        """Test that load_tracking with XML returns a TrackingDataset object."""
        from fastforward._dataset import TrackingDataset

        result = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_XML_PATH, lazy=False
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
        return statsperform.load_tracking(MA25_DATA_PATH, MA1_JSON_PATH, lazy=False)

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
        """Test provider is 'statsperform'."""
        assert metadata_df["provider"][0] == "statsperform"

    def test_coordinate_system_value(self, metadata_df):
        """Test coordinate_system is 'cdf' (default)."""
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_orientation_value(self, metadata_df):
        """Test orientation is 'static_home_away'."""
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_team_names(self, metadata_df):
        """Test team names are extracted correctly."""
        assert metadata_df["home_team"][0] == "Monaco"
        assert metadata_df["away_team"][0] == "Reims"

    def test_team_ids(self, metadata_df):
        """Test team IDs are extracted correctly."""
        assert metadata_df["home_team_id"][0] == "4t4hod56fsj7utpjdor8so5q6"
        assert metadata_df["away_team_id"][0] == "3c3jcs7vc1t6vz5lev162jyv7"

    def test_pitch_dimensions(self, metadata_df):
        """Test pitch dimensions are correct (105m x 68m default)."""
        pitch_length = metadata_df["pitch_length"][0]
        pitch_width = metadata_df["pitch_width"][0]

        assert pitch_length == pytest.approx(105.0, rel=0.01)
        assert pitch_width == pytest.approx(68.0, rel=0.01)

    def test_fps(self, metadata_df):
        """Test fps is 10 (100ms frame rate)."""
        assert metadata_df["fps"][0] == pytest.approx(10.0, rel=0.01)

    def test_game_id(self, metadata_df):
        """Test game_id is present."""
        assert metadata_df["game_id"][0] == "7ijuqohwgmplbxdj1625sxwfe"

    def test_game_date(self, metadata_df):
        """Test game date (2020-08-23 Monaco vs Reims)."""
        from datetime import date

        assert metadata_df["game_date"][0] == date(2020, 8, 23)


class TestTeamDataFrame:
    """Tests for the team DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return statsperform.load_tracking(MA25_DATA_PATH, MA1_JSON_PATH, lazy=False)

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
        assert home_team["name"][0] == "Monaco"
        assert away_team["name"][0] == "Reims"


class TestPlayerDataFrame:
    """Tests for the player DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return statsperform.load_tracking(MA25_DATA_PATH, MA1_JSON_PATH, lazy=False)

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
        """Test that player_df contains 40 players (20+20 from squads + subs)."""
        # Monaco has 20 players, Reims has 20 players
        assert player_df.height == 40

    def test_home_players(self, player_df):
        """Test home team (Monaco) players."""
        home_players = player_df.filter(
            pl.col("team_id") == "4t4hod56fsj7utpjdor8so5q6"
        )
        assert home_players.height == 20

    def test_away_players(self, player_df):
        """Test away team (Reims) players."""
        away_players = player_df.filter(
            pl.col("team_id") == "3c3jcs7vc1t6vz5lev162jyv7"
        )
        assert away_players.height == 20

    def test_specific_home_player(self, player_df):
        """Test specific home player (Wissam Ben Yedder)."""
        ben_yedder = player_df.filter(pl.col("player_id") == "a2s2c6anax9wnlsw1s6vunl5h")
        assert ben_yedder.height == 1
        assert ben_yedder["name"][0] == "Wissam Ben Yedder"
        assert ben_yedder["team_id"][0] == "4t4hod56fsj7utpjdor8so5q6"
        assert ben_yedder["jersey_number"][0] == 9
        assert ben_yedder["position"][0] == "ST"

    def test_specific_away_player(self, player_df):
        """Test specific away player (Boulaye Dia)."""
        dia = player_df.filter(pl.col("player_id") == "3mqhkh4iz4kesvmqui3hupgmi")
        assert dia.height == 1
        assert dia["name"][0] == "Boulaye Dia"
        assert dia["team_id"][0] == "3c3jcs7vc1t6vz5lev162jyv7"
        assert dia["jersey_number"][0] == 11
        assert dia["position"][0] == "RW"

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
        """Load tracking data with long layout."""
        return statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            layout="long",
            only_alive=False,
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

    def test_has_ball_rows(self, tracking_df):
        """Test that long format includes ball as separate rows."""
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        # One ball row per frame (93 frames total)
        assert ball_rows.height == 93

        # Check ball rows have player_id = "ball"
        assert all(pid == "ball" for pid in ball_rows["player_id"].to_list())

    def test_timestamp_type(self, tracking_df):
        """Test that timestamp is Duration type."""
        assert tracking_df.schema["timestamp"] == pl.Duration("ms")

    def test_has_two_periods(self, tracking_df):
        """Test that data includes 2 periods (period 1 and period 2)."""
        periods = tracking_df["period_id"].unique().sort().to_list()
        assert len(periods) == 2
        assert periods == [1, 2]

    def test_frame_count(self, tracking_df):
        """Test total number of unique frames."""
        unique_frames = tracking_df["frame_id"].n_unique()
        # Test file: 26 frames in period 1, 41 alive frames in period 2 (67 total with only_alive=True)
        # Note: Frame IDs restart per period, so unique count is period 2's count
        assert unique_frames == 67


class TestTrackingDataFrameLongBall:
    """Tests for tracking DataFrame with 'long_ball' layout."""

    @pytest.fixture
    def dataset(self):
        """Load tracking data with long_ball layout."""
        return statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
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
        """Load tracking data with wide layout."""
        return statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            layout="wide",
            only_alive=False,
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

        # 23 unique players appear in tracking data
        assert len(x_columns) == 23
        assert len(y_columns) == 23
        assert len(z_columns) == 23

    def test_one_row_per_frame(self, tracking_df):
        """Test that wide format has exactly one row per frame."""
        # 93 total frames in the test file (26 in period 1, 67 in period 2)
        # Note: frame_ids restart per period, so we check total row count
        assert tracking_df.height == 93


class TestCoordinateSystem:
    """Tests for coordinate system parameter."""

    def test_cdf_coordinates_default(self):
        """Test that CDF coordinates are used by default."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            lazy=False,
        )
        assert dataset.tracking.height == 2117
        assert dataset.metadata["coordinate_system"][0] == "cdf"

    def test_statsperform_coordinates(self):
        """Test that statsperform coordinates work correctly."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            coordinates="statsperform",
            lazy=False,
        )
        assert dataset.tracking.height == 2117
        assert dataset.metadata["coordinate_system"][0] == "statsperform"

    def test_sportvu_coordinates_alias(self):
        """Test that 'sportvu' is an alias for statsperform coordinates."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            coordinates="sportvu",
            lazy=False,
        )
        assert dataset.tracking.height == 2117
        assert dataset.metadata["coordinate_system"][0] == "sportvu"


class TestLayoutParameter:
    """Tests for layout parameter validation."""

    def test_invalid_layout(self):
        """Test that invalid layout raises error."""
        with pytest.raises(Exception):
            statsperform.load_tracking(
                MA25_DATA_PATH,
                MA1_JSON_PATH,
                layout="invalid",
                lazy=False,
            )


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_tracking_file(self):
        """Test that missing tracking file raises error."""
        with pytest.raises(Exception):
            statsperform.load_tracking(
                "nonexistent_tracking.txt",
                MA1_JSON_PATH,
                lazy=False,
            )

    def test_missing_metadata_file(self):
        """Test that missing metadata file raises error."""
        with pytest.raises(Exception):
            statsperform.load_tracking(
                MA25_DATA_PATH,
                "nonexistent_metadata.json",
                lazy=False,
            )


class TestOnlyAliveParameter:
    """Tests for only_alive parameter."""

    def test_only_alive_filters_dead_frames(self):
        """Test that only_alive=True filters out dead ball frames."""
        dataset_all = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            only_alive=False,
            lazy=False,
        )
        dataset_alive = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            only_alive=True,
            lazy=False,
        )

        # Test data has no dead frames, so both should have same count
        assert dataset_all.tracking.height == 2117
        assert dataset_alive.tracking.height == 2117

    def test_only_alive_no_dead_frames(self):
        """Test that only_alive=True results in no dead ball frames."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
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
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, lazy=False
        )
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_orientation_attack_left(self):
        """Test orientation 'attack_left'."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            orientation="attack_left",
            lazy=False,
        )
        assert dataset.metadata["orientation"][0] == "attack_left"


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyParameter:
    """Tests for lazy loading parameter."""

    def test_lazy_false_returns_dataframe(self):
        """Test that lazy=False returns a DataFrame."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, lazy=False
        )
        assert isinstance(dataset.tracking, pl.DataFrame)

    def test_lazy_true_returns_lazyframe(self):
        """Test that lazy=True returns a LazyFrame."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, lazy=True
        )
        assert isinstance(dataset.tracking, pl.LazyFrame)

    def test_lazy_collect_works(self):
        """Test that collecting a LazyFrame works."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, lazy=True
        )
        collected = dataset.tracking.collect()
        assert isinstance(collected, pl.DataFrame)
        assert collected.height == 2117


class TestLazyNotImplemented:
    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            statsperform.load_tracking(MA25_DATA_PATH, MA1_JSON_PATH, lazy=True)


class TestIncludeGameId:
    """Tests for include_game_id parameter."""

    def test_include_game_id_true(self):
        """Test that include_game_id=True adds game_id column."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            include_game_id=True,
            lazy=False,
        )
        assert "game_id" in dataset.tracking.columns
        assert dataset.tracking["game_id"][0] == "7ijuqohwgmplbxdj1625sxwfe"

    def test_include_game_id_false(self):
        """Test that include_game_id=False removes game_id column."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            include_game_id=False,
            lazy=False,
        )
        assert "game_id" not in dataset.tracking.columns

    def test_include_game_id_custom_string(self):
        """Test that include_game_id=str uses custom value."""
        custom_id = "custom_game_123"
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            include_game_id=custom_id,
            lazy=False,
        )
        assert "game_id" in dataset.tracking.columns
        assert dataset.tracking["game_id"][0] == custom_id


class TestStatsPerformCoordinates:
    """Tests specific to StatsPerform coordinate handling."""

    @pytest.fixture
    def dataset_cdf(self):
        """Load and return the dataset with CDF coordinates."""
        return statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            coordinates="cdf",
            only_alive=False,
            lazy=False,
        )

    @pytest.fixture
    def dataset_native(self):
        """Load and return the dataset with native StatsPerform coordinates.

        Uses static_away_home orientation to avoid coordinate flipping,
        allowing verification of raw native coordinate values.
        """
        return statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            coordinates="statsperform",
            orientation="static_away_home",
            only_alive=False,
            lazy=False,
        )

    def test_cdf_coordinates_center_origin(self, dataset_cdf):
        """Test that CDF coordinates have center origin."""
        tracking_df = dataset_cdf.tracking
        player_data = tracking_df.filter(pl.col("team_id") != "ball")

        x_values = player_data["x"].to_list()
        y_values = player_data["y"].to_list()

        # Check we have both positive and negative x values (center origin)
        has_positive_x = any(x > 0 for x in x_values if x is not None)
        has_negative_x = any(x < 0 for x in x_values if x is not None)
        assert has_positive_x and has_negative_x

        # Check we have both positive and negative y values
        has_positive_y = any(y > 0 for y in y_values if y is not None)
        has_negative_y = any(y < 0 for y in y_values if y is not None)
        assert has_positive_y and has_negative_y

    def test_native_coordinates_top_left_origin(self, dataset_native):
        """Test that native StatsPerform coordinates have top-left origin."""
        tracking_df = dataset_native.tracking
        player_data = tracking_df.filter(pl.col("team_id") != "ball")

        x_values = player_data["x"].to_list()
        y_values = player_data["y"].to_list()

        # All values should be positive (top-left origin)
        assert all(x >= 0 for x in x_values if x is not None)
        assert all(y >= 0 for y in y_values if y is not None)

    def test_ball_coordinates_first_frame(self, dataset_cdf):
        """Test ball coordinates are parsed and transformed correctly (first frame)."""
        tracking_df = dataset_cdf.tracking
        ball_data = tracking_df.filter(pl.col("team_id") == "ball")

        # First frame ball position
        first_frame = ball_data.filter(pl.col("frame_id") == 0).row(0, named=True)

        # Original: 52.35, 33.25 (SportVU top-left origin)
        # CDF conversion: x = 52.35 - 52.5 = -0.15, y = 34 - 33.25 = 0.75
        # Orientation: Home attacks left (detected from player positions), so
        # for static_home_away (home attacks right), coordinates are flipped:
        # Final: x = 0.15, y = -0.75
        assert first_frame["x"] == pytest.approx(0.15, abs=0.5)
        assert first_frame["y"] == pytest.approx(-0.75, abs=0.5)
        assert first_frame["z"] == pytest.approx(0.0, abs=0.01)

    def test_goalkeeper_position(self, dataset_native):
        """Test goalkeeper positions are parsed correctly in native coords."""
        tracking_df = dataset_native.tracking

        # Check home goalkeeper (Monaco GK jersey 40, player_id=6wfwy94p5bm0zv3aku0urfq39)
        first_frame = tracking_df.filter(
            (pl.col("frame_id") == 0) & (pl.col("player_id") == "6wfwy94p5bm0zv3aku0urfq39")
        )
        if first_frame.height > 0:
            row = first_frame.row(0, named=True)
            # First frame: 100.245, 32.224 - near goal line (x ~105 would be goal)
            assert row["x"] == pytest.approx(100.245, abs=0.01)
            assert row["y"] == pytest.approx(32.224, abs=0.01)


class TestPeriodsDataFrame:
    """Tests for the periods DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return statsperform.load_tracking(MA25_DATA_PATH, MA1_JSON_PATH, lazy=False)

    @pytest.fixture
    def periods_df(self, dataset):
        """Load and return the periods DataFrame."""
        return dataset.periods

    def test_has_periods(self, periods_df):
        """Test that periods_df has two periods."""
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

    def test_first_period_frames(self, periods_df):
        """Test first period frame IDs."""
        period_1 = periods_df.filter(pl.col("period_id") == 1)
        assert period_1["start_frame_id"][0] == 0
        # Period 1 has 26 frames (0-25)
        assert period_1["end_frame_id"][0] == 25

    def test_period_timing(self, periods_df):
        """Test that all periods have correct timing values."""
        from datetime import timedelta

        periods = periods_df.sort("period_id")

        # Period 1
        p1 = periods.row(0, named=True)
        assert p1["start_timestamp"] == timedelta(milliseconds=0)
        assert p1["end_timestamp"] == timedelta(milliseconds=2500)
        assert p1["duration"] == timedelta(milliseconds=2500)

        # Period 2
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] == timedelta(milliseconds=0)
        assert p2["end_timestamp"] == timedelta(milliseconds=6600)
        assert p2["duration"] == timedelta(milliseconds=6600)


class TestPitchDimensions:
    """Tests for pitch dimension parameters."""

    def test_custom_pitch_dimensions(self):
        """Test custom pitch dimensions are applied."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH,
            MA1_JSON_PATH,
            pitch_length=110.0,
            pitch_width=75.0,
            lazy=False,
        )
        assert dataset.metadata["pitch_length"][0] == pytest.approx(110.0, rel=0.01)
        assert dataset.metadata["pitch_width"][0] == pytest.approx(75.0, rel=0.01)


class TestMetadataAutoDetection:
    """Tests for auto-detection of metadata format (JSON vs XML)."""

    def test_json_metadata_detection(self):
        """Test that JSON metadata is correctly detected and parsed."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, lazy=False
        )
        # Basic check that it parsed
        assert dataset.metadata["game_id"][0] == "7ijuqohwgmplbxdj1625sxwfe"

    def test_xml_metadata_detection(self):
        """Test that XML metadata is correctly detected and parsed."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_XML_PATH, lazy=False
        )
        # Basic check that it parsed
        assert dataset.metadata["game_id"][0] == "7ijuqohwgmplbxdj1625sxwfe"


class TestOfficials:
    """Tests for include_officials parameter."""

    def test_include_officials_false_by_default(self):
        """Officials not included by default."""
        dataset = statsperform.load_tracking(MA25_DATA_PATH, MA1_JSON_PATH, lazy=False)
        officials = dataset.players.filter(pl.col("team_id") == "officials")
        assert officials.height == 0

    def test_include_officials_true_json(self):
        """Officials included when requested (JSON metadata)."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, include_officials=True, lazy=False
        )
        officials = dataset.players.filter(pl.col("team_id") == "officials")
        # Main referee, 2 assistants, 4th official
        assert officials.height == 4

    def test_include_officials_true_xml(self):
        """Officials included when requested (XML metadata)."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_XML_PATH, include_officials=True, lazy=False
        )
        officials = dataset.players.filter(pl.col("team_id") == "officials")
        # Main referee, 2 assistants, 4th official
        assert officials.height == 4

    def test_official_positions(self):
        """Officials have correct position codes."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, include_officials=True, lazy=False
        )
        officials = dataset.players.filter(pl.col("team_id") == "officials")
        positions = set(officials["position"].to_list())
        # REF (Main), AREF (2x Assistant referee), 4TH (Fourth official)
        assert positions == {"REF", "AREF", "4TH"}

    def test_officials_team_exists(self):
        """Officials team added when include_officials=True."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, include_officials=True, lazy=False
        )
        teams = dataset.teams["team_id"].to_list()
        assert "officials" in teams

    def test_officials_team_not_exists_by_default(self):
        """Officials team not present by default."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, include_officials=False, lazy=False
        )
        teams = dataset.teams["team_id"].to_list()
        assert "officials" not in teams

    def test_main_referee_details(self):
        """Test main referee has correct details."""
        dataset = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, include_officials=True, lazy=False
        )
        main_ref = dataset.players.filter(
            (pl.col("team_id") == "officials") & (pl.col("position") == "REF")
        )
        assert main_ref.height == 1
        assert main_ref["player_id"][0] == "1ch8qs93kraoj5m3l1xfdytw5"
        assert main_ref["name"][0] == "François Letexier"

    def test_player_count_with_officials(self):
        """Test total player count includes officials."""
        dataset_without = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, include_officials=False, lazy=False
        )
        dataset_with = statsperform.load_tracking(
            MA25_DATA_PATH, MA1_JSON_PATH, include_officials=True, lazy=False
        )
        # 40 players + 4 officials = 44
        assert dataset_without.players.height == 40
        assert dataset_with.players.height == 44


class TestBomHandling:
    """Tests for UTF-8 BOM handling in StatsPerform XML metadata."""

    def test_bom_xml_metadata_loads(self):
        """Test that BOM-prefixed XML metadata loads without error."""
        dataset = statsperform.load_tracking(MA25_DATA_PATH, MA1_XML_BOM_PATH, lazy=False)
        assert dataset.tracking.height > 0

    def test_bom_xml_matches_non_bom(self):
        """Test that BOM-prefixed XML metadata produces same results as non-BOM."""
        dataset_normal = statsperform.load_tracking(MA25_DATA_PATH, MA1_XML_PATH, only_alive=False, lazy=False)
        dataset_bom = statsperform.load_tracking(MA25_DATA_PATH, MA1_XML_BOM_PATH, only_alive=False, lazy=False)

        assert dataset_bom.tracking.height == dataset_normal.tracking.height
        assert dataset_bom.players.height == dataset_normal.players.height
