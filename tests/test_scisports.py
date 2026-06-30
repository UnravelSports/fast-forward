"""Integration tests for SciSports EPTS tracking data loading."""

import pytest
import polars as pl

from fastforward import scisports
from tests.config import SCI_META_XML, SCI_RAW_TXT


class TestLoadTracking:
    """Tests for scisports.load_tracking function."""

    def test_returns_dataset(self):
        from fastforward import TrackingDataset

        dataset = scisports.load_tracking(SCI_RAW_TXT, SCI_META_XML, lazy=False)

        assert isinstance(dataset, TrackingDataset)
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert isinstance(dataset.metadata, pl.DataFrame)
        assert isinstance(dataset.teams, pl.DataFrame)
        assert isinstance(dataset.players, pl.DataFrame)
        assert isinstance(dataset.periods, pl.DataFrame)


class TestMetadataDataFrame:
    """Tests for the metadata DataFrame."""

    @pytest.fixture(scope="class")
    def dataset(self):
        return scisports.load_tracking(SCI_RAW_TXT, SCI_META_XML, lazy=False)

    def test_pitch_dimensions(self, dataset):
        meta = dataset.metadata
        assert meta["pitch_length"][0] == pytest.approx(105.0, rel=1e-3)
        assert meta["pitch_width"][0] == pytest.approx(68.0, rel=1e-3)

    def test_fps(self, dataset):
        assert dataset.metadata["fps"][0] == pytest.approx(25.0, rel=1e-3)

    def test_provider(self, dataset):
        assert dataset.metadata["provider"][0] == "scisports"

    def test_team_names(self, dataset):
        # Kloppy PR fixture has placeholder names.
        assert dataset.metadata["home_team"][0] == "Home Team"
        assert dataset.metadata["away_team"][0] == "Away Team"


class TestPeriodsDataFrame:

    @pytest.fixture(scope="class")
    def dataset(self):
        return scisports.load_tracking(SCI_RAW_TXT, SCI_META_XML, lazy=False)

    def test_two_periods(self, dataset):
        assert dataset.periods.height == 2
        assert sorted(dataset.periods["period_id"].to_list()) == [1, 2]


class TestPlayersDataFrame:

    @pytest.fixture(scope="class")
    def dataset(self):
        return scisports.load_tracking(SCI_RAW_TXT, SCI_META_XML, lazy=False)

    def test_thirty_two_players(self, dataset):
        # Kloppy PR fixture: 32 players total across both teams.
        assert dataset.players.height == 32

    def test_has_jersey_numbers(self, dataset):
        # All players should have a non-null jersey_number.
        assert dataset.players["jersey_number"].null_count() == 0

    def test_subs_marked_as_non_starters(self, dataset):
        # Per the new-provider-checklist bug fix: per-player Start Frame > first
        # period's Start Frame ⇒ sub (is_starter=False). The kloppy PR sets
        # everyone to True.
        players = dataset.players
        # At least some subs in this fixture.
        n_starters = players.filter(pl.col("is_starter")).height
        n_subs = players.filter(~pl.col("is_starter")).height
        assert n_starters > 0
        assert n_subs > 0, "expected some subs in the fixture"


class TestTrackingDataFrame:

    @pytest.fixture(scope="class")
    def dataset(self):
        return scisports.load_tracking(
            SCI_RAW_TXT, SCI_META_XML, lazy=False, only_alive=False
        )

    def test_has_expected_columns(self, dataset):
        # Standard "long" layout columns from the shared dataframe builder.
        for col in ("frame_id", "period_id", "team_id", "player_id", "x", "y"):
            assert col in dataset.tracking.columns, f"missing {col}"

    def test_periods_present(self, dataset):
        seen = sorted(set(dataset.tracking["period_id"].to_list()))
        assert seen == [1, 2]

    def test_no_frames_outside_periods(self, dataset):
        # Pre-kickoff (frame_id < period 1 start_frame) is filtered at parse.
        first_period_start_frame = (
            dataset.periods.filter(pl.col("period_id") == 1)["start_frame_id"][0]
        )
        n_before = dataset.tracking.filter(
            pl.col("frame_id") < first_period_start_frame
        ).height
        assert n_before == 0


class TestSpecificFrame:
    """Validate the y/x swap by checking a specific player position the kloppy
    PR pins. The kloppy test asserts the home GK is at (x=8.36, y=34.17) at
    frame 44659 (the very first in-period frame)."""

    @pytest.fixture(scope="class")
    def dataset(self):
        return scisports.load_tracking(
            SCI_RAW_TXT,
            SCI_META_XML,
            lazy=False,
            only_alive=False,
            coordinates="sportvu",  # un-rotated, top-left meters — matches kloppy's reference frame
            orientation="home_away",  # half-by-half, matching kloppy default
        )

    def test_gk_at_first_frame(self, dataset):
        first_frame = dataset.tracking.filter(pl.col("frame_id") == 44659)
        assert first_frame.height > 0, "expected frame 44659 in fixture"

        # The GK is the player with jersey 1 on the home team. We don't have a
        # direct GK marker on the row, but jersey_number is on the players_df.
        # Find via the player_id of jersey-1 home player.
        home_gk = dataset.players.filter(pl.col("jersey_number") == 1).head(1)
        assert home_gk.height > 0
        home_gk_id = home_gk["player_id"][0]

        gk_row = first_frame.filter(pl.col("player_id") == home_gk_id)
        assert gk_row.height == 1, f"GK {home_gk_id} not at frame 44659"
        # Kloppy PR asserts (x=8.36, y=34.17) here. After our swap we expect
        # the same.
        assert gk_row["x"][0] == pytest.approx(8.36, abs=0.1)
        assert gk_row["y"][0] == pytest.approx(34.17, abs=0.1)


class TestBallStateFiltering:

    def test_only_alive_default_filters(self):
        ds_all = scisports.load_tracking(
            SCI_RAW_TXT, SCI_META_XML, lazy=False, only_alive=False
        )
        ds_alive = scisports.load_tracking(
            SCI_RAW_TXT, SCI_META_XML, lazy=False, only_alive=True
        )
        # only_alive should drop strictly fewer rows than the full dataset.
        assert ds_alive.tracking.height <= ds_all.tracking.height
        # Confirm there are some dead frames in the fixture.
        assert ds_alive.tracking.height < ds_all.tracking.height


class TestGameId:

    def test_default_includes_game_id_from_metadata(self):
        ds = scisports.load_tracking(SCI_RAW_TXT, SCI_META_XML, lazy=False)
        assert "game_id" in ds.tracking.columns
        # game_id is the Session id of the Full Match session in the EPTS XML.
        assert ds.metadata["game_id"][0] == "1527118440"

    def test_string_overrides(self):
        ds = scisports.load_tracking(
            SCI_RAW_TXT, SCI_META_XML, lazy=False, include_game_id="my_match_42"
        )
        assert set(ds.tracking["game_id"].unique().to_list()) == {"my_match_42"}

    def test_false_omits(self):
        ds = scisports.load_tracking(
            SCI_RAW_TXT, SCI_META_XML, lazy=False, include_game_id=False
        )
        assert "game_id" not in ds.tracking.columns


class TestSchemasFactory:

    def test_factory_matches_dataset_schema(self):
        from fastforward._schemas import Schemas

        # arrow[spark] is bytes-only by contract — read the fixtures first.
        with open(SCI_RAW_TXT, "rb") as f:
            raw = f.read()
        with open(SCI_META_XML, "rb") as f:
            meta = f.read()

        ds = scisports.load_tracking(raw, meta, lazy=False, engine="arrow[spark]")
        factory = scisports.schemas(layout="long", engine="arrow[spark]")
        assert isinstance(factory, Schemas)
        assert factory.tracking == ds.tracking.schema
