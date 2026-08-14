"""Integration tests for SkillCorner tracking data loading."""

import pytest
import polars as pl

from fastforward import skillcorner
from tests.config import (
    SC_RAW as RAW_DATA_PATH,
    SC_META as META_DATA_PATH,
)


class TestLoadTracking:
    """Tests for skillcorner.load_tracking function."""

    def test_returns_dataset(self):
        """Test that load_tracking returns a TrackingDataset."""
        from fastforward import TrackingDataset

        dataset = skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

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
        return skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

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
    def dataset(self):
        """Load and return the dataset."""
        return skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def team_df(self, dataset):
        """Return the team DataFrame."""
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
        return skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def player_df(self, dataset):
        """Return the player DataFrame."""
        return dataset.players

    def test_schema(self, player_df):
        """Test that player_df has expected columns."""
        expected_columns = {"game_id", "team_id", "player_id", "name", "nickname", "first_name", "last_name", "jersey_number", "position", "is_starter", "minutes_played", "minutes_ball_in_play", "minutes_in_possession", "minutes_out_possession"}
        assert set(player_df.columns) == expected_columns

    def test_has_players(self, player_df):
        """Test that player_df contains players from both teams."""
        assert player_df.height == 36

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


class TestPeriodsDataFrame:
    """Tests for the periods DataFrame."""

    @pytest.fixture
    def dataset(self):
        """Load and return the dataset."""
        return skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

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
        assert p1["end_timestamp"] == timedelta(milliseconds=8100)
        assert p1["duration"] == timedelta(milliseconds=8100)

        # Period 2
        p2 = periods.row(1, named=True)
        assert p2["start_timestamp"] == timedelta(milliseconds=0)
        assert p2["end_timestamp"] == timedelta(milliseconds=6500)
        assert p2["duration"] == timedelta(milliseconds=6500)


class TestTrackingDataFrameLong:
    """Tests for tracking DataFrame with 'long' layout."""

    @pytest.fixture
    def dataset(self):
        """Load dataset with long layout."""
        return skillcorner.load_tracking(
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
            "ball_owning_player_id",
            "team_id",
            "player_id",
            "x",
            "y",
            "z",
            "is_detected",
        }
        assert set(tracking_df.columns) == expected_columns

    def test_has_ball_rows(self, tracking_df):
        """Test that long format includes ball as separate rows."""
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 148

        # Check ball rows have player_id = "ball"
        assert ball_rows["player_id"].to_list()[:10] == ["ball"] * 10

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
        return skillcorner.load_tracking(
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
            "ball_owning_player_id",
            "ball_x",
            "ball_y",
            "ball_z",
            "team_id",
            "player_id",
            "x",
            "y",
            "z",
            "is_detected",
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
        return skillcorner.load_tracking(
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

        assert len(x_columns) == 22
        assert len(x_columns) == len(y_columns) == len(z_columns)

    def test_one_row_per_frame(self, tracking_df):
        """Test that wide format has exactly one row per frame."""
        frame_count = tracking_df["frame_id"].n_unique()
        assert tracking_df.height == frame_count


class TestIncludeEmptyFrames:
    """Tests for include_empty_frames parameter."""

    def test_empty_frames_excluded_by_default(self):
        """Test that empty frames are excluded by default."""
        dataset_no_empty = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, include_empty_frames=False, lazy=False
        )
        dataset_with_empty = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, include_empty_frames=True, lazy=False
        )

        # With empty frames should have more rows (or equal if no empty frames)
        assert dataset_with_empty.tracking.height >= dataset_no_empty.tracking.height


class TestOnlyAliveParameter:
    """Tests for only_alive parameter."""

    def test_only_alive_filters_dead_frames(self):
        """Test that only_alive=True filters out dead ball frames."""
        dataset_all = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=False, lazy=False
        )
        dataset_alive = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True, lazy=False
        )

        # alive has fewer rows than all (some dead frames excluded)
        assert dataset_all.tracking.height == 4600
        assert dataset_alive.tracking.height == 3404

    def test_only_alive_no_dead_frames(self):
        """Test that only_alive=True results in no dead ball frames."""
        dataset = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, only_alive=True, lazy=False
        )

        # All rows should be alive
        dead_rows = dataset.tracking.filter(pl.col("ball_state") == "dead")
        assert dead_rows.height == 0


class TestOrientationParameter:
    """Tests for orientation parameter."""

    def test_orientation_default_static_home_away(self):
        """Test that orientation defaults to 'static_home_away'."""
        dataset = skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_orientation_static_away_home(self):
        """Test that orientation='static_away_home' is recorded."""
        dataset = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "static_away_home"

    def test_invalid_orientation(self):
        """Test that invalid orientation raises error."""
        with pytest.raises(Exception):
            skillcorner.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH, orientation="invalid", lazy=False
            )


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_tracking_file(self):
        """Test that missing tracking file raises error."""
        with pytest.raises(Exception):
            skillcorner.load_tracking("nonexistent_tracking.jsonl", META_DATA_PATH, lazy=False)

    def test_missing_metadata_file(self):
        """Test that missing metadata file raises error."""
        with pytest.raises(Exception):
            skillcorner.load_tracking(RAW_DATA_PATH, "nonexistent_metadata.json", lazy=False)


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyParameter:
    """Tests for lazy loading parameter."""

    def test_lazy_returns_lazyframe(self):
        """Test that lazy=True returns a TrackingDataset with pl.LazyFrame."""
        from fastforward import TrackingDataset

        dataset = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.LazyFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)  # Metadata is eager
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)

    def test_lazy_collect_returns_dataframe(self):
        """Test that collect() returns a DataFrame."""
        dataset = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        result = dataset.tracking.collect()
        assert isinstance(result, pl.DataFrame)

    def test_lazy_filter_before_collect(self):
        """Test that filter() can be chained before collect()."""
        dataset = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        result = dataset.tracking.filter(pl.col("period_id") == 1).collect()
        # All rows should be period 1
        assert all(p == 1 for p in result["period_id"].to_list())

    def test_lazy_collect_matches_eager(self):
        """Test that lazy collect() produces same result as eager loading."""
        dataset_lazy = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=True
        )
        dataset_eager = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, lazy=False
        )
        assert dataset_lazy.tracking.collect().equals(dataset_eager.tracking)


class TestLazyNotImplemented:
    def test_lazy_raises(self):
        with pytest.raises(NotImplementedError, match="lazy loading"):
            skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

    def test_from_cache_raises(self):
        with pytest.raises(NotImplementedError, match="cache loading"):
            skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, from_cache=True)


class TestTimestampBehavior:
    """Tests for timestamp and FPS behavior."""

    @pytest.fixture
    def dataset(self):
        """Load dataset."""
        return skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    @pytest.fixture
    def tracking_df(self, dataset):
        """Return tracking DataFrame."""
        return dataset.tracking

    @pytest.fixture
    def metadata_df(self, dataset):
        """Return metadata DataFrame."""
        return dataset.metadata

    def test_period_1_first_frame_timestamp_near_zero(self, tracking_df):
        """Test that period 1 first frame timestamp is at or near 0ms."""
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        if len(period_1) > 0:
            first_timestamp = period_1["timestamp"][0]
            # Should be at 0ms or very close (within first 200ms for 10fps data)
            assert first_timestamp.total_seconds() * 1000 < 200

    def test_period_2_first_frame_timestamp_near_zero(self, tracking_df):
        """Test that period 2 first frame timestamp is at or near 0ms (period-relative)."""
        period_2 = tracking_df.filter(pl.col("period_id") == 2).sort("frame_id")
        if len(period_2) > 0:
            first_timestamp = period_2["timestamp"][0]
            # Should be at 0ms or very close (within first 200ms for 10fps data)
            # NOT 45 minutes (old cumulative timestamp behavior)
            assert first_timestamp.total_seconds() * 1000 < 200

    def test_timestamp_matches_fps(self, tracking_df, metadata_df):
        """Test that timestamp increments match the FPS (10fps = 100ms per frame)."""
        fps = metadata_df["fps"][0]
        expected_delta_ms = 1000 / fps  # 100ms for 10fps

        # Get first few frames of period 1 with unique frame_ids
        period_1 = tracking_df.filter(pl.col("period_id") == 1).sort("frame_id")
        unique_frames = period_1.unique(subset=["frame_id"], maintain_order=True).head(5)

        if len(unique_frames) >= 2:
            ts1 = unique_frames["timestamp"][0].total_seconds() * 1000
            ts2 = unique_frames["timestamp"][1].total_seconds() * 1000
            delta = ts2 - ts1
            # Allow some tolerance (within 20ms)
            assert abs(delta - expected_delta_ms) < 20

    def test_metadata_fps_value(self, metadata_df):
        """Test that FPS is correctly reported as 10."""
        assert metadata_df["fps"][0] == pytest.approx(10.0, rel=0.01)


# ----------------------------------------------------------------------------
# Expected values precomputed from the fixture by an offline developer tool
# (scripts/precompute_skillcorner_extras_expected.py, gitignored so it doesn't
# ship). The script parses the fixture without importing fastforward, so the
# values below are an independent reference, not a self-consistency check.
#
# Regenerate after any fixture change:
#     python scripts/precompute_skillcorner_extras_expected.py
# ----------------------------------------------------------------------------

EXPECTED_ROSTER_SIZE = 36
EXPECTED_ALIVE_FRAMES = 148
EXPECTED_FRAMES_WITH_POSSESSION_PLAYER = 36
# All possession.player_id values in the fixture must resolve to a roster
# player. If this drops below the total, the fixture has the partial-
# anonymization bug back (stale possession refs to ids outside the roster).
EXPECTED_FRAMES_WITH_RESOLVED_POSSESSION = 36

EXPECTED_LONG_ROWS = 3404
EXPECTED_LONG_BALL_ROWS = 3256
EXPECTED_BALL_ROWS = 148

EXPECTED_IS_DETECTED_TRUE = 2862
EXPECTED_IS_DETECTED_FALSE = 394
EXPECTED_IS_DETECTED_NULL_LONG = 148  # one per ball row

EXPECTED_BALL_OWNING_PLAYER_NON_NULL_LONG = 828
EXPECTED_BALL_OWNING_PLAYER_NON_NULL_LONG_BALL = 792
EXPECTED_BALL_OWNING_PLAYER_NON_NULL_WIDE = 36


class TestSkillCornerExtras:
    """Tests for `include_ball_owning_player` and `include_is_detected`, which
    default to True (both columns present unless explicitly disabled)."""

    # ---- warning behaviour --------------------------------------------------

    def test_default_no_future_warning(self, recwarn):
        """The default (both flags on) must not emit a FutureWarning."""
        skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        future_warnings = [w for w in recwarn.list if issubclass(w.category, FutureWarning)]
        assert future_warnings == []

    def test_explicit_false_no_warning(self, recwarn):
        skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=False,
            include_is_detected=False,
            lazy=False,
        )
        future_warnings = [w for w in recwarn.list if issubclass(w.category, FutureWarning)]
        assert future_warnings == []

    def test_explicit_true_no_warning(self, recwarn):
        skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=True,
            include_is_detected=True,
            lazy=False,
        )
        future_warnings = [w for w in recwarn.list if issubclass(w.category, FutureWarning)]
        assert future_warnings == []

    # ---- column presence / absence ------------------------------------------

    def test_default_includes_extras_columns(self):
        ds = skillcorner.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert "ball_owning_player_id" in ds.tracking.columns
        assert "is_detected" in ds.tracking.columns

    def test_include_ball_owning_player_adds_column(self):
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=True,
            include_is_detected=False,
            lazy=False,
        )
        assert "ball_owning_player_id" in ds.tracking.columns

    def test_include_is_detected_adds_column_long(self):
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=False,
            include_is_detected=True,
            layout="long",
            lazy=False,
        )
        assert "is_detected" in ds.tracking.columns

    def test_include_is_detected_adds_column_long_ball(self):
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=False,
            include_is_detected=True,
            layout="long_ball",
            lazy=False,
        )
        assert "is_detected" in ds.tracking.columns

    def test_include_is_detected_wide_is_noop_for_now(self):
        # Wide layout would need a per-player column; not implemented in this
        # initial cut. The flag is accepted but doesn't add columns. Document
        # the limitation by asserting the absence.
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=False,
            include_is_detected=True,
            layout="wide",
            lazy=False,
        )
        assert "is_detected" not in ds.tracking.columns

    # ---- exact value checks (precomputed independently) ---------------------

    def test_long_row_count(self):
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=True,
            include_is_detected=True,
            lazy=False,
        )
        assert ds.tracking.height == EXPECTED_LONG_ROWS

    def test_long_ball_row_count(self):
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=True,
            include_is_detected=True,
            layout="long_ball",
            lazy=False,
        )
        assert ds.tracking.height == EXPECTED_LONG_BALL_ROWS

    def test_is_detected_distribution_long(self):
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=False,
            include_is_detected=True,
            layout="long",
            lazy=False,
        )
        true_n = ds.tracking.filter(pl.col("is_detected") == True).height  # noqa: E712
        false_n = ds.tracking.filter(pl.col("is_detected") == False).height  # noqa: E712
        null_n = ds.tracking.filter(pl.col("is_detected").is_null()).height
        assert true_n == EXPECTED_IS_DETECTED_TRUE
        assert false_n == EXPECTED_IS_DETECTED_FALSE
        # In long layout the ball row has null is_detected by design.
        assert null_n == EXPECTED_IS_DETECTED_NULL_LONG

    def test_is_detected_distribution_long_ball(self):
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=False,
            include_is_detected=True,
            layout="long_ball",
            lazy=False,
        )
        true_n = ds.tracking.filter(pl.col("is_detected") == True).height  # noqa: E712
        false_n = ds.tracking.filter(pl.col("is_detected") == False).height  # noqa: E712
        # long_ball has no ball row, so no nulls from that source.
        null_n = ds.tracking.filter(pl.col("is_detected").is_null()).height
        assert true_n == EXPECTED_IS_DETECTED_TRUE
        assert false_n == EXPECTED_IS_DETECTED_FALSE
        assert null_n == 0

    def test_ball_owning_player_non_null_counts(self):
        for layout, expected in [
            ("long", EXPECTED_BALL_OWNING_PLAYER_NON_NULL_LONG),
            ("long_ball", EXPECTED_BALL_OWNING_PLAYER_NON_NULL_LONG_BALL),
            ("wide", EXPECTED_BALL_OWNING_PLAYER_NON_NULL_WIDE),
        ]:
            ds = skillcorner.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH,
                include_ball_owning_player=True,
                include_is_detected=False,
                layout=layout,
                lazy=False,
            )
            n = ds.tracking.filter(pl.col("ball_owning_player_id").is_not_null()).height
            assert n == expected, f"{layout} layout: expected {expected}, got {n}"

    # ---- correctness invariants (catch fixture / parser drift) --------------

    def test_owning_player_ids_are_real_players(self):
        """Every non-null ball_owning_player_id must match a row in
        ds.players. Designed to catch the class of bug where a fixture has
        stale possession.player_id values that don't resolve to the roster
        (the previous partial-anonymization issue)."""
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=True,
            include_is_detected=False,
            lazy=False,
        )
        roster_ids = set(ds.players["player_id"].to_list())
        owners = (
            ds.tracking.filter(pl.col("ball_owning_player_id").is_not_null())
            ["ball_owning_player_id"]
            .unique()
            .to_list()
        )
        # The fixture must have at least one possession-bearing frame, and
        # every distinct owner UUID must be in the roster.
        assert len(owners) > 0, "fixture has no non-null possession rows"
        unknown = [uid for uid in owners if uid not in roster_ids]
        assert not unknown, (
            f"unknown ball_owning_player_id values: {unknown}. "
            "Likely cause: the fixture has stale possession.player_id refs "
            "that don't match the roster's player.id values."
        )

    def test_owning_player_consistent_per_frame(self):
        """All rows of a given (period_id, frame_id) in long layout must
        share the same ball_owning_player_id value (or be null together).
        Tests that the broadcast join is correct."""
        ds = skillcorner.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            include_ball_owning_player=True,
            include_is_detected=False,
            lazy=False,
        )
        per_frame = (
            ds.tracking.group_by("period_id", "frame_id")
            .agg(pl.col("ball_owning_player_id").n_unique().alias("distinct"))
        )
        # n_unique counts non-null distinct values: 0 (no owner) or 1 (one
        # owner broadcast to all rows). Never > 1.
        assert per_frame["distinct"].max() <= 1
