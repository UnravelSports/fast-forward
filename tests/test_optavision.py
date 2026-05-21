"""Integration tests for OptaVision (StatsPerform FIFA EPTS) tracking loader."""

import datetime

import polars as pl
import pytest

from fastforward import optavision
from fastforward._dataset import TrackingDataset
from tests.config import OV_META as META_PATH, OV_RAW as RAW_PATH

GAME_ID = "swyrn1j6ovvowhnpeuza3xu27"
HOME_TEAM_ID = "hpvjeond44ey0isf0arwr9w6d"
AWAY_TEAM_ID = "sxfdkddi5le7xqgqwxosc5evs"
EXPECTED_PITCH_LENGTH = 105.0
EXPECTED_PITCH_WIDTH = 68.0
EXPECTED_FPS = 25.0
EXPECTED_GAME_DATE = datetime.date(2026, 2, 28)
EXPECTED_PLAYER_COUNT = 39
EXPECTED_STARTERS_PER_TEAM = 11

EXPECTED_P1_START, EXPECTED_P1_END = 10000, 10099
EXPECTED_P2_START, EXPECTED_P2_END = 1000000, 1000099
EXPECTED_PERIOD_FRAME_SPAN = 99      
EXPECTED_PERIOD_DURATION_MS = 3960    

EXPECTED_LONG_ROWS = 4600      
EXPECTED_LONG_BALL_ROWS = 4400  
EXPECTED_WIDE_ROWS = 200       
EXPECTED_BALL_ROWS = 200        

EXPECTED_P1_HOME_MEAN_X = -14.598236
EXPECTED_P1_AWAY_MEAN_X = 8.576109

EXPECTED_P2_HOME_MEAN_X_STATIC = -8.452918
EXPECTED_P2_AWAY_MEAN_X_STATIC = 12.399136
EXPECTED_P2_HOME_MEAN_X_HOME_AWAY = 8.452918
EXPECTED_P2_AWAY_MEAN_X_HOME_AWAY = -12.399136

EXPECTED_MAX_ABS_X = 46.2483
EXPECTED_MAX_ABS_Y = 27.7435
EXPECTED_BALL_MEAN_X = -4.528502
EXPECTED_BALL_MEAN_Y = 1.232922

EXPECTED_GK_PLAYER_ID = "15c30g67fwca723zlgk67agbb"
EXPECTED_GK_X = -45.8136
EXPECTED_GK_Y = 0.5066

EXPECTED_POSSESSION_FRAMES = 102           
EXPECTED_POSSESSION_NONNULL_ROWS = 2346   

EXPECTED_BALL_F10000_X = -0.8984
EXPECTED_BALL_F10000_Y = -2.5633
EXPECTED_BALL_F10000_Z = 0.0

FLOAT_TOL = 1e-3


class TestLoadTracking:
    """Smoke tests for the top-level load_tracking entry point."""

    def test_returns_dataset(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)
        assert isinstance(ds, TrackingDataset)
        assert isinstance(ds.tracking, pl.DataFrame)
        assert isinstance(ds.metadata, pl.DataFrame)
        assert isinstance(ds.teams, pl.DataFrame)
        assert isinstance(ds.players, pl.DataFrame)
        assert isinstance(ds.periods, pl.DataFrame)

    def test_swapped_args_raises(self):
        with pytest.raises(Exception):
            optavision.load_tracking(META_PATH, RAW_PATH, lazy=False)

    def test_missing_tracking_file_raises(self):
        with pytest.raises(Exception):
            optavision.load_tracking("/nonexistent/file.txt", META_PATH, lazy=False)

    def test_missing_metadata_file_raises(self):
        with pytest.raises(Exception):
            optavision.load_tracking(RAW_PATH, "/nonexistent/file.xml", lazy=False)


class TestMetadataDataFrame:
    @pytest.fixture
    def metadata_df(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, lazy=False).metadata

    def test_single_row(self, metadata_df):
        assert metadata_df.height == 1

    def test_schema(self, metadata_df):
        expected = {
            "provider", "game_id", "game_date",
            "home_team", "home_team_id", "away_team", "away_team_id",
            "pitch_length", "pitch_width", "fps",
            "coordinate_system", "orientation",
        }
        assert set(metadata_df.columns) == expected

    def test_provider_value(self, metadata_df):
        assert metadata_df["provider"][0] == "optavision"

    def test_game_id_from_match_uuid(self, metadata_df):
        assert metadata_df["game_id"][0] == GAME_ID

    def test_pitch_dimensions(self, metadata_df):
        assert metadata_df["pitch_length"][0] == EXPECTED_PITCH_LENGTH
        assert metadata_df["pitch_width"][0] == EXPECTED_PITCH_WIDTH

    def test_fps(self, metadata_df):
        assert metadata_df["fps"][0] == EXPECTED_FPS

    def test_coordinate_system_default(self, metadata_df):
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_orientation_default(self, metadata_df):
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_team_ids(self, metadata_df):
        assert metadata_df["home_team_id"][0] == HOME_TEAM_ID
        assert metadata_df["away_team_id"][0] == AWAY_TEAM_ID

    def test_game_date(self, metadata_df):
        assert metadata_df["game_date"][0] == datetime.date(2026, 2, 28)


class TestTeamDataFrame:
    @pytest.fixture
    def team_df(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, lazy=False).teams

    def test_two_rows(self, team_df):
        assert team_df.height == 2

    def test_schema(self, team_df):
        assert set(team_df.columns) == {"game_id", "team_id", "name", "ground"}

    def test_home_and_away(self, team_df):
        assert set(team_df["ground"].to_list()) == {"home", "away"}

    def test_first_listed_team_is_home(self, team_df):
        home_row = team_df.filter(pl.col("ground") == "home").row(0, named=True)
        assert home_row["team_id"] == HOME_TEAM_ID


class TestPlayerDataFrame:
    @pytest.fixture
    def player_df(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, lazy=False).players

    def test_schema(self, player_df):
        expected = {
            "game_id", "team_id", "player_id", "name",
            "first_name", "last_name", "jersey_number", "position", "is_starter",
        }
        assert set(player_df.columns) == expected

    def test_total_player_count(self, player_df):
        assert player_df.height == EXPECTED_PLAYER_COUNT

    def test_eleven_starters_per_team(self, player_df):
        # Determined from the first frame of period 1.
        starters = player_df.filter(pl.col("is_starter")).group_by("team_id").len()
        for row in starters.iter_rows(named=True):
            assert row["len"] == EXPECTED_STARTERS_PER_TEAM, (
                f"team {row['team_id']} has {row['len']} starters"
            )

    def test_positions_are_standard_codes(self, player_df):
        valid = {
            "GK", "LB", "RB", "CB", "LCB", "RCB", "LWB", "RWB",
            "LDM", "CDM", "RDM", "LCM", "CM", "RCM", "LAM", "CAM", "RAM",
            "LM", "RM", "LW", "RW", "LF", "ST", "RF", "CF",
            "SUB", "UNK", "REF", "AREF", "VAR", "AVAR", "4TH",
        }
        assert set(player_df["position"].to_list()).issubset(valid)


class TestPeriodsDataFrame:
    @pytest.fixture
    def periods_df(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, lazy=False).periods

    def test_two_periods(self, periods_df):
        assert periods_df.height == 2

    def test_schema(self, periods_df):
        expected = {
            "game_id", "period_id",
            "start_frame_id", "end_frame_id",
            "start_timestamp", "end_timestamp", "duration",
        }
        assert set(periods_df.columns) == expected

    def test_frame_boundaries_match_fixture(self, periods_df):
        rows = list(periods_df.sort("period_id").iter_rows(named=True))
        assert rows[0]["start_frame_id"] == EXPECTED_P1_START
        assert rows[0]["end_frame_id"] == EXPECTED_P1_END
        assert rows[1]["start_frame_id"] == EXPECTED_P2_START
        assert rows[1]["end_frame_id"] == EXPECTED_P2_END
        for row in rows:
            assert row["end_frame_id"] - row["start_frame_id"] == EXPECTED_PERIOD_FRAME_SPAN


class TestTrackingDataFrameLong:
    @pytest.fixture
    def tracking_df(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, layout="long", lazy=False).tracking

    def test_schema(self, tracking_df):
        # Default include_ball_owning_player=True adds the extra column.
        expected = {
            "game_id", "frame_id", "period_id", "timestamp",
            "ball_state", "ball_owning_team_id",
            "team_id", "player_id", "x", "y", "z",
            "ball_owning_player_id",
        }
        assert set(tracking_df.columns) == expected

    def test_has_ball_rows(self, tracking_df):
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == EXPECTED_BALL_ROWS

    def test_total_long_row_count(self, tracking_df):
        assert tracking_df.height == EXPECTED_LONG_ROWS

    def test_timestamp_is_duration_ms(self, tracking_df):
        assert tracking_df.schema["timestamp"] == pl.Duration("ms")

    def test_both_periods_present(self, tracking_df):
        assert set(tracking_df["period_id"].unique().to_list()) == {1, 2}

    def test_all_ball_states_alive(self, tracking_df):
        # OptaVision exports only contain in-play frames; every row must be alive.
        assert set(tracking_df["ball_state"].unique().to_list()) == {"alive"}


class TestTrackingDataFrameLongBall:
    @pytest.fixture
    def tracking_df(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, layout="long_ball", lazy=False).tracking

    def test_schema(self, tracking_df):
        expected = {
            "game_id", "frame_id", "period_id", "timestamp",
            "ball_state", "ball_owning_team_id",
            "team_id", "player_id", "x", "y", "z",
            "ball_x", "ball_y", "ball_z",
            "ball_owning_player_id",
        }
        assert set(tracking_df.columns) == expected

    def test_no_ball_rows(self, tracking_df):
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 0

    def test_total_long_ball_row_count(self, tracking_df):
        assert tracking_df.height == EXPECTED_LONG_BALL_ROWS


class TestTrackingDataFrameWide:
    @pytest.fixture
    def tracking_df(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, layout="wide", lazy=False).tracking

    def test_one_row_per_frame(self, tracking_df):
        assert tracking_df.height == EXPECTED_WIDE_ROWS

    def test_base_columns_present(self, tracking_df):
        base = {"game_id", "frame_id", "period_id", "timestamp", "ball_state", "ball_owning_team_id"}
        assert base.issubset(set(tracking_df.columns))


class TestOnlyAliveIsNoOp:
    """OptaVision exports only contain in-play frames; only_alive has no effect."""

    def test_only_alive_true_equals_false(self):
        ds_true = optavision.load_tracking(RAW_PATH, META_PATH, only_alive=True, lazy=False)
        ds_false = optavision.load_tracking(RAW_PATH, META_PATH, only_alive=False, lazy=False)
        assert ds_true.tracking.equals(ds_false.tracking)


class TestOrientation:
    def test_default_orientation_in_metadata(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)
        assert ds.metadata["orientation"][0] == "static_home_away"

    def test_static_away_home_recorded(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, orientation="static_away_home", lazy=False)
        assert ds.metadata["orientation"][0] == "static_away_home"

    def test_invalid_orientation_raises(self):
        with pytest.raises(Exception):
            optavision.load_tracking(RAW_PATH, META_PATH, orientation="not_a_real_orientation", lazy=False)

    def test_static_home_away_keeps_home_left_in_period_1(self):
        # Period 1 metadata: home team plays LeftToRight, so under static_home_away
        # (home always attacks right) coordinates should NOT be flipped, and home's
        # mean x at frame 10000 should match the raw value parsed from the fixture.
        ds = optavision.load_tracking(RAW_PATH, META_PATH, orientation="static_home_away", lazy=False)
        home_x = (
            ds.tracking
            .filter(pl.col("frame_id") == EXPECTED_P1_START)
            .filter(pl.col("team_id") == HOME_TEAM_ID)["x"]
            .mean()
        )
        assert home_x == pytest.approx(EXPECTED_P1_HOME_MEAN_X, abs=FLOAT_TOL)


class TestDirectionsOfPlay:
    """Verify that <DirectionsOfPlay> from the metadata is honoured."""

    @pytest.fixture
    def dataset(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)

    def test_period_1_home_attacks_right(self, dataset):
        # Metadata: home plays LeftToRight in P1. Under static_home_away (default)
        # P1 isn't flipped, so the per-team means must match the raw fixture values.
        first_frame = dataset.tracking.filter(pl.col("frame_id") == EXPECTED_P1_START)
        home_avg_x = first_frame.filter(pl.col("team_id") == HOME_TEAM_ID)["x"].mean()
        away_avg_x = first_frame.filter(pl.col("team_id") == AWAY_TEAM_ID)["x"].mean()
        assert home_avg_x == pytest.approx(EXPECTED_P1_HOME_MEAN_X, abs=FLOAT_TOL)
        assert away_avg_x == pytest.approx(EXPECTED_P1_AWAY_MEAN_X, abs=FLOAT_TOL)

    def test_period_2_under_home_away_orientation(self):
        # home_away preserves the natural half-time side swap (home attacks +x in
        # odd periods, -x in even). Both periods' detected directions already
        # match the target, so P2 is NOT flipped; coords equal the raw values.
        ds = optavision.load_tracking(RAW_PATH, META_PATH, orientation="home_away", lazy=False)
        first_frame = ds.tracking.filter(pl.col("frame_id") == EXPECTED_P2_START)
        home_avg_x = first_frame.filter(pl.col("team_id") == HOME_TEAM_ID)["x"].mean()
        away_avg_x = first_frame.filter(pl.col("team_id") == AWAY_TEAM_ID)["x"].mean()
        assert home_avg_x == pytest.approx(EXPECTED_P2_HOME_MEAN_X_HOME_AWAY, abs=FLOAT_TOL)
        assert away_avg_x == pytest.approx(EXPECTED_P2_AWAY_MEAN_X_HOME_AWAY, abs=FLOAT_TOL)

    def test_period_2_under_static_home_away_keeps_home_left(self, dataset):
        # Under static_home_away (home attacks +x in both periods), P2 IS flipped
        # because raw P2 has home attacking RtL. After flipping, the per-team
        # means are the negation of the raw P2 values.
        first_frame = dataset.tracking.filter(pl.col("frame_id") == EXPECTED_P2_START)
        home_avg_x = first_frame.filter(pl.col("team_id") == HOME_TEAM_ID)["x"].mean()
        away_avg_x = first_frame.filter(pl.col("team_id") == AWAY_TEAM_ID)["x"].mean()
        assert home_avg_x == pytest.approx(EXPECTED_P2_HOME_MEAN_X_STATIC, abs=FLOAT_TOL)
        assert away_avg_x == pytest.approx(EXPECTED_P2_AWAY_MEAN_X_STATIC, abs=FLOAT_TOL)


class TestCoordinateSystem:
    """Verify that the parser produces CDF-aligned coordinates."""

    @pytest.fixture
    def dataset(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)

    def test_x_extends_to_far_end_of_pitch(self, dataset):
        max_abs_x = dataset.tracking["x"].abs().max()
        assert max_abs_x == pytest.approx(EXPECTED_MAX_ABS_X, abs=FLOAT_TOL)

    def test_y_extends_to_far_side_of_pitch(self, dataset):
        max_abs_y = dataset.tracking["y"].abs().max()
        assert max_abs_y == pytest.approx(EXPECTED_MAX_ABS_Y, abs=FLOAT_TOL)

    def test_origin_is_centred(self, dataset):
        ball = dataset.tracking.filter(pl.col("team_id") == "ball")
        assert ball["x"].mean() == pytest.approx(EXPECTED_BALL_MEAN_X, abs=FLOAT_TOL)
        assert ball["y"].mean() == pytest.approx(EXPECTED_BALL_MEAN_Y, abs=FLOAT_TOL)

    def test_x_axis_sign_matches_directions_of_play(self, dataset):
        first_frame = dataset.tracking.filter(pl.col("frame_id") == EXPECTED_P1_START)
        home_avg_x = first_frame.filter(pl.col("team_id") == HOME_TEAM_ID)["x"].mean()
        assert home_avg_x == pytest.approx(EXPECTED_P1_HOME_MEAN_X, abs=FLOAT_TOL)

    def test_home_goalkeeper_position(self, dataset):
        first_frame = dataset.tracking.filter(pl.col("frame_id") == EXPECTED_P1_START)
        home_players = first_frame.filter(pl.col("team_id") == HOME_TEAM_ID).sort("x")
        gk = home_players.row(0, named=True)
        assert gk["player_id"] == EXPECTED_GK_PLAYER_ID
        assert gk["x"] == pytest.approx(EXPECTED_GK_X, abs=FLOAT_TOL)
        assert gk["y"] == pytest.approx(EXPECTED_GK_Y, abs=FLOAT_TOL)


class TestBallOwningPlayer:
    """Verify the OptaVision-specific include_ball_owning_player flag."""

    def test_default_includes_column(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)
        assert "ball_owning_player_id" in ds.tracking.columns

    def test_disabled_omits_column(self):
        ds = optavision.load_tracking(
            RAW_PATH, META_PATH, include_ball_owning_player=False, lazy=False
        )
        assert "ball_owning_player_id" not in ds.tracking.columns

    def test_non_null_row_count(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)
        non_null = ds.tracking.filter(pl.col("ball_owning_player_id").is_not_null()).height
        assert non_null == EXPECTED_POSSESSION_NONNULL_ROWS

    def test_owning_player_ids_are_real_players(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)
        roster_ids = set(ds.players["player_id"].to_list())
        owners = (
            ds.tracking.filter(pl.col("ball_owning_player_id").is_not_null())
            ["ball_owning_player_id"]
            .unique()
            .to_list()
        )
        unknown = [uid for uid in owners if uid not in roster_ids]
        assert not unknown, f"unknown owning-player UUIDs: {unknown}"

    def test_owning_player_consistent_per_frame(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)
        per_frame = (
            ds.tracking.group_by("period_id", "frame_id")
            .agg(pl.col("ball_owning_player_id").n_unique().alias("distinct"))
        )
        assert per_frame["distinct"].max() == 1
        assert per_frame["distinct"].min() == 1

    def test_works_across_layouts(self):
        for layout in ("long", "long_ball", "wide"):
            ds = optavision.load_tracking(RAW_PATH, META_PATH, layout=layout, lazy=False)
            assert "ball_owning_player_id" in ds.tracking.columns, (
                f"missing ball_owning_player_id in {layout} layout"
            )


class TestTimestampBehavior:
    @pytest.fixture
    def dataset(self):
        return optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)

    def test_period_1_timestamp_range(self, dataset):
        p1 = dataset.tracking.filter(pl.col("period_id") == 1)
        assert p1["timestamp"].min().total_seconds() * 1000 == 0
        assert p1["timestamp"].max().total_seconds() * 1000 == EXPECTED_PERIOD_DURATION_MS

    def test_period_2_timestamp_range(self, dataset):
        p2 = dataset.tracking.filter(pl.col("period_id") == 2)
        assert p2["timestamp"].min().total_seconds() * 1000 == 0
        assert p2["timestamp"].max().total_seconds() * 1000 == EXPECTED_PERIOD_DURATION_MS


class TestGameIdControl:
    def test_default_uses_match_uuid(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)
        assert ds.metadata["game_id"][0] == GAME_ID
        assert ds.tracking["game_id"][0] == GAME_ID

    def test_false_omits_game_id(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, include_game_id=False, lazy=False)
        assert "game_id" not in ds.tracking.columns

    def test_str_overrides_game_id(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, include_game_id="custom-id", lazy=False)
        assert ds.tracking["game_id"][0] == "custom-id"


class TestLazyNotImplemented:
    """Lazy loading and from_cache are not yet supported; they must raise."""

    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            optavision.load_tracking(RAW_PATH, META_PATH, lazy=True)

    def test_from_cache_raises(self):
        with pytest.raises(NotImplementedError, match="cache"):
            optavision.load_tracking(RAW_PATH, META_PATH, from_cache=True)


class TestRoundTrip:
    """Cross-check: ball coordinates in the parsed DF match the raw file."""

    def test_first_frame_ball_position(self):
        ds = optavision.load_tracking(RAW_PATH, META_PATH, lazy=False)
        ball_row = (
            ds.tracking
            .filter(pl.col("frame_id") == EXPECTED_P1_START)
            .filter(pl.col("team_id") == "ball")
            .row(0, named=True)
        )
        assert ball_row["x"] == pytest.approx(EXPECTED_BALL_F10000_X, abs=FLOAT_TOL)
        assert ball_row["y"] == pytest.approx(EXPECTED_BALL_F10000_Y, abs=FLOAT_TOL)
        assert ball_row["z"] == EXPECTED_BALL_F10000_Z
