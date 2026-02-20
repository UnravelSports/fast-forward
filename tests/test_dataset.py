"""Tests for TrackingDataset class."""

import pytest
import polars as pl

from fastforward import secondspectrum
from fastforward._dataset import TrackingDataset
from tests.config import (
    DATA_DIR,
    SS_RAW_ANON as RAW_DATA_PATH,
    SS_META_ANON as META_DATA_PATH,
)


class TestTrackingDatasetStructure:
    """Tests for TrackingDataset basic structure."""

    def test_returns_tracking_dataset_instance(self):
        """Test that load_tracking returns TrackingDataset instance."""
        result = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert isinstance(result, TrackingDataset)

    def test_has_tracking_property(self):
        """Test that TrackingDataset has tracking property."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert hasattr(dataset, 'tracking')
        assert isinstance(dataset.tracking, pl.DataFrame)

    def test_has_metadata_property(self):
        """Test that TrackingDataset has metadata property."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert hasattr(dataset, 'metadata')
        assert isinstance(dataset.metadata, pl.DataFrame)

    def test_has_teams_property(self):
        """Test that TrackingDataset has teams property."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert hasattr(dataset, 'teams')
        assert isinstance(dataset.teams, pl.DataFrame)

    def test_has_players_property(self):
        """Test that TrackingDataset has players property."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert hasattr(dataset, 'players')
        assert isinstance(dataset.players, pl.DataFrame)

    def test_has_periods_property(self):
        """Test that TrackingDataset has periods property."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert hasattr(dataset, 'periods')
        assert isinstance(dataset.periods, pl.DataFrame)


class TestTrackingDatasetEager:
    """Tests for eager loading (lazy=False)."""

    def test_eager_tracking_is_dataframe(self):
        """Test that eager tracking is a DataFrame."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert len(dataset.tracking) == 4554

    def test_eager_metadata_loaded(self):
        """Test that eager metadata is loaded."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert dataset.metadata.height == 1

    def test_eager_teams_loaded(self):
        """Test that eager teams are loaded."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert dataset.teams.height == 2

    def test_eager_players_loaded(self):
        """Test that eager players are loaded."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert dataset.players.height == 40


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestTrackingDatasetLazy:
    """Tests for lazy loading (lazy=True)."""

    def test_lazy_tracking_is_lazyframe(self):
        """Test that lazy tracking is a pl.LazyFrame."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        assert isinstance(dataset.tracking, pl.LazyFrame)

    def test_lazy_metadata_is_eager(self):
        """Test that metadata is always eager."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        assert isinstance(dataset.metadata, pl.DataFrame)
        assert dataset.metadata.height == 1

    def test_lazy_teams_is_eager(self):
        """Test that teams are always eager."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        assert isinstance(dataset.teams, pl.DataFrame)
        assert dataset.teams.height == 2

    def test_lazy_players_is_eager(self):
        """Test that players are always eager."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        assert isinstance(dataset.players, pl.DataFrame)
        assert dataset.players.height == 40

    def test_lazy_can_collect(self):
        """Test that lazy tracking can be collected."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        result = dataset.tracking.collect()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 4554

    def test_lazy_collect_equals_eager(self):
        """Test that lazy collect equals eager loading."""
        dataset_lazy = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        dataset_eager = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        assert dataset_lazy.tracking.collect().equals(dataset_eager.tracking)


class TestPeriodsDataFrame:
    """Tests for periods DataFrame property."""

    @pytest.fixture
    def dataset(self):
        """Load dataset."""
        return secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

    def test_periods_df_exists(self, dataset):
        """Test that periods DataFrame exists."""
        assert hasattr(dataset, 'periods')
        assert isinstance(dataset.periods, pl.DataFrame)

    def test_periods_schema(self, dataset):
        """Test that periods DataFrame has correct schema."""
        # game_id is included by default (include_game_id=True)
        expected_columns = {"game_id", "period_id", "start_frame_id", "end_frame_id", "start_timestamp", "end_timestamp", "duration"}
        assert set(dataset.periods.columns) == expected_columns

    def test_periods_has_rows(self, dataset):
        """Test that periods DataFrame has rows."""
        assert dataset.periods.height == 2

    def test_periods_period_ids(self, dataset):
        """Test that period IDs are sequential starting from 1."""
        period_ids = dataset.periods["period_id"].sort().to_list()
        assert period_ids[0] == 1
        assert period_ids[1] == 2

    def test_periods_frame_ids_valid(self, dataset):
        """Test that start and end frame IDs are valid."""
        for row in dataset.periods.iter_rows(named=True):
            assert row["start_frame_id"] is not None
            assert row["end_frame_id"] is not None
            assert row["start_frame_id"] < row["end_frame_id"]

    def test_periods_have_valid_frame_ranges(self, dataset):
        """Test that periods DataFrame has valid frame ranges."""
        periods = dataset.periods

        # Check that all periods have end > start
        for row in periods.iter_rows(named=True):
            assert row["end_frame_id"] > row["start_frame_id"], \
                f"Period {row['period_id']} has invalid frame range"

        # Check period 1 and 2 exist
        period_1 = periods.filter(pl.col("period_id") == 1)
        period_2 = periods.filter(pl.col("period_id") == 2)
        assert period_1.height == 1
        assert period_2.height == 1

    def test_periods_only_includes_existing_periods(self, dataset):
        """Test that periods only includes periods with data."""
        # Test data only has 2 periods
        assert dataset.periods.height == 2

        # Period 3 should not be in periods DataFrame
        period_3 = dataset.periods.filter(pl.col("period_id") == 3)
        assert period_3.height == 0


class TestTrackingDatasetRepr:
    """Tests for __repr__ method."""

    def test_repr_contains_game_id(self):
        """Test that repr contains game_id."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        repr_str = repr(dataset)
        assert "game_id=" in repr_str

    @pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
    def test_repr_contains_tracking_type(self):
        """Test that repr shows tracking type."""
        # Eager loading
        dataset_eager = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        repr_eager = repr(dataset_eager)
        assert "tracking=DataFrame" in repr_eager

        # Lazy loading
        dataset_lazy = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)
        repr_lazy = repr(dataset_lazy)
        assert "tracking=LazyFrame" in repr_lazy

    def test_repr_contains_periods_count(self):
        """Test that repr contains periods count."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        repr_str = repr(dataset)
        assert "periods=2" in repr_str

    def test_repr_contains_players_count(self):
        """Test that repr contains players count."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        repr_str = repr(dataset)
        assert "players=" in repr_str

    def test_repr_format(self):
        """Test overall repr format."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)
        repr_str = repr(dataset)
        # Should start with TrackingDataset(
        assert repr_str.startswith("TrackingDataset(")
        assert repr_str.endswith(")")


class TestTrackingDatasetConsistency:
    """Tests for data consistency across properties."""

    def test_game_id_consistent(self):
        """Test that game_id is consistent across all DataFrames."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

        game_id = dataset.metadata["game_id"][0]

        # Tracking
        tracking_game_ids = dataset.tracking["game_id"].unique()
        assert len(tracking_game_ids) == 1
        assert tracking_game_ids[0] == game_id

        # Teams
        team_game_ids = dataset.teams["game_id"].unique()
        assert len(team_game_ids) == 1
        assert team_game_ids[0] == game_id

        # Players
        player_game_ids = dataset.players["game_id"].unique()
        assert len(player_game_ids) == 1
        assert player_game_ids[0] == game_id

    def test_team_ids_consistent(self):
        """Test that team_ids in teams and players match."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

        team_ids_from_teams = set(dataset.teams["team_id"].to_list())
        team_ids_from_players = set(dataset.players["team_id"].to_list())

        assert team_ids_from_players.issubset(team_ids_from_teams)

    def test_player_ids_in_tracking(self):
        """Test that player_ids in tracking exist in players DataFrame."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=False)

        player_ids_from_players = set(dataset.players["player_id"].to_list())
        player_ids_from_tracking = set(
            dataset.tracking.filter(pl.col("team_id") != "ball")["player_id"].unique().to_list()
        )

        # Tracking may have fewer players (not all players appear in tracking)
        assert player_ids_from_tracking.issubset(player_ids_from_players)


class TestTrackingDatasetWithDifferentProviders:
    """Test TrackingDataset works with all providers."""

    def test_skillcorner_returns_dataset(self):
        """Test that SkillCorner returns TrackingDataset."""
        from fastforward import skillcorner

        sc_raw = str(DATA_DIR / "skillcorner_tracking.jsonl")
        sc_meta = str(DATA_DIR / "skillcorner_meta.json")

        dataset = skillcorner.load_tracking(sc_raw, sc_meta, lazy=False)
        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.periods, pl.DataFrame)

    def test_sportec_returns_dataset(self):
        """Test that Sportec returns TrackingDataset."""
        from fastforward import sportec

        sp_raw = str(DATA_DIR / "sportec_positional.xml")
        sp_meta = str(DATA_DIR / "sportec_meta.xml")

        dataset = sportec.load_tracking(sp_raw, sp_meta, lazy=False)
        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.periods, pl.DataFrame)

    def test_hawkeye_returns_dataset(self):
        """Test that HawkEye returns TrackingDataset."""
        from fastforward import hawkeye

        ball_files = [
            str(DATA_DIR / "hawkeye_1_1.football.samples.ball"),
            str(DATA_DIR / "hawkeye_2_46.football.samples.ball"),
        ]
        player_files = [
            str(DATA_DIR / "hawkeye_1_1.football.samples.centroids"),
            str(DATA_DIR / "hawkeye_2_46.football.samples.centroids"),
        ]
        meta_json = str(DATA_DIR / "hawkeye_meta.json")

        dataset = hawkeye.load_tracking(ball_files, player_files, meta_json, lazy=False)
        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.periods, pl.DataFrame)


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyFrameFunctionality:
    """Tests for full pl.LazyFrame functionality with lazy loading."""

    def test_schema_access_before_collect(self):
        """Test that schema is accessible before calling collect()."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

        # Schema should be accessible without loading data
        schema = dataset.tracking.collect_schema()
        assert "frame_id" in schema
        assert "period_id" in schema
        assert "x" in schema
        assert "y" in schema

    def test_with_columns_operation(self):
        """Test that with_columns works on lazy tracking."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

        # Add a new column
        result = (
            dataset.tracking
            .with_columns((pl.col("x") * 100).alias("x_cm"))
            .collect()
        )

        assert "x_cm" in result.columns
        # x_cm should be approximately 100x the x value (allow for float precision)
        sample = result.select(["x", "x_cm"]).head(1)
        assert abs(sample["x_cm"][0] - sample["x"][0] * 100) < 0.01

    def test_group_by_operation(self):
        """Test that group_by works on lazy tracking."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

        # Group by player and compute mean x
        result = (
            dataset.tracking
            .filter(pl.col("team_id") != "ball")
            .group_by("player_id")
            .agg(pl.col("x").mean().alias("mean_x"))
            .collect()
        )

        assert "player_id" in result.columns
        assert "mean_x" in result.columns
        assert len(result) > 0

    def test_join_operation(self):
        """Test that join works between lazy tracking and players."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

        # Join tracking with players to get player names
        result = (
            dataset.tracking
            .filter(pl.col("team_id") != "ball")
            .head(100)
            .join(
                dataset.players.lazy(),
                on="player_id",
                how="left"
            )
            .collect()
        )

        # Should have player columns from the join
        assert "player_id" in result.columns
        # Original tracking columns should still be present
        assert "x" in result.columns
        assert "frame_id" in result.columns

    def test_filter_multiple_conditions(self):
        """Test complex filtering on lazy tracking."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

        result = (
            dataset.tracking
            .filter(
                (pl.col("period_id") == 1) &
                (pl.col("team_id") != "ball") &
                (pl.col("x") > 0)
            )
            .collect()
        )

        # All rows should match filters
        assert len(result) == 770
        assert all(p == 1 for p in result["period_id"].to_list())
        assert "ball" not in result["team_id"].to_list()
        assert all(x > 0 for x in result["x"].to_list())

    def test_sort_operation(self):
        """Test that sort works on lazy tracking."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

        result = (
            dataset.tracking
            .sort("frame_id", descending=True)
            .head(10)
            .collect()
        )

        # Should be sorted descending by frame_id
        frame_ids = result["frame_id"].to_list()
        assert frame_ids == sorted(frame_ids, reverse=True)

    def test_unique_operation(self):
        """Test that unique works on lazy tracking."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

        result = (
            dataset.tracking
            .select("team_id")
            .unique()
            .collect()
        )

        # Should have unique team_ids including "ball"
        team_ids = result["team_id"].to_list()
        assert "ball" in team_ids
        # At most 3 unique values (home, away, ball)
        assert len(team_ids) <= 3

    def test_lazy_chain_multiple_operations(self):
        """Test chaining multiple lazy operations."""
        dataset = secondspectrum.load_tracking(RAW_DATA_PATH, META_DATA_PATH, lazy=True)

        result = (
            dataset.tracking
            .filter(pl.col("period_id") == 1)
            .filter(pl.col("team_id") != "ball")
            .with_columns((pl.col("x") ** 2 + pl.col("y") ** 2).sqrt().alias("distance"))
            .group_by("player_id")
            .agg([
                pl.col("distance").mean().alias("avg_distance"),
                pl.col("x").count().alias("n_observations")
            ])
            .sort("avg_distance", descending=True)
            .collect()
        )

        assert "player_id" in result.columns
        assert "avg_distance" in result.columns
        assert "n_observations" in result.columns
        # Should be sorted by avg_distance descending
        distances = result["avg_distance"].to_list()
        assert distances == sorted(distances, reverse=True)
