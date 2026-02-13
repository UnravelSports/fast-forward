"""Tests for include_game_id parameter across all providers."""

import pytest
import polars as pl

from fastforward import secondspectrum, skillcorner, sportec
from tests.config import (
    SS_RAW_ANON as SS_RAW,
    SS_META_ANON as SS_META,
    SC_RAW,
    SC_META,
    SP_RAW,
    SP_META,
)


class TestIncludeGameIdSecondSpectrum:
    """Tests for include_game_id parameter with SecondSpectrum provider."""

    # Test include_game_id=True (default)
    def test_true_tracking_df_has_game_id(self):
        """When include_game_id=True, tracking_df should have game_id column."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True, lazy=False)
        t = dataset.tracking
        assert "game_id" in t.columns
        assert t["game_id"][0] is not None

    def test_true_team_df_has_game_id(self):
        """When include_game_id=True, team_df should have game_id column."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True, lazy=False)
        team = dataset.teams
        assert "game_id" in team.columns
        assert team["game_id"][0] is not None

    def test_true_player_df_has_game_id(self):
        """When include_game_id=True, player_df should have game_id column."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True, lazy=False)
        player = dataset.players
        assert "game_id" in player.columns
        assert player["game_id"][0] is not None

    def test_true_metadata_df_has_game_id(self):
        """When include_game_id=True, metadata_df should have game_id column."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True, lazy=False)
        m = dataset.metadata
        assert "game_id" in m.columns
        assert m["game_id"][0] is not None

    # Test include_game_id=False
    def test_false_tracking_df_no_game_id_column(self):
        """When include_game_id=False, tracking_df should not have game_id column."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=False, lazy=False)
        t = dataset.tracking
        assert "game_id" not in t.columns

    def test_false_team_df_no_game_id_column(self):
        """When include_game_id=False, team_df should not have game_id column."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=False, lazy=False)
        team = dataset.teams
        assert "game_id" not in team.columns

    def test_false_player_df_no_game_id_column(self):
        """When include_game_id=False, player_df should not have game_id column."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=False, lazy=False)
        player = dataset.players
        assert "game_id" not in player.columns

    def test_false_metadata_df_still_has_game_id(self):
        """When include_game_id=False, metadata_df should still have game_id column."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=False, lazy=False)
        m = dataset.metadata
        assert "game_id" in m.columns
        assert m["game_id"][0] is not None

    # Test include_game_id="custom_string"
    def test_string_tracking_df_has_custom_game_id(self):
        """When include_game_id is a string, tracking_df should use that value."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id="custom_id_123", lazy=False)
        t = dataset.tracking
        assert t["game_id"][0] == "custom_id_123"

    def test_string_team_df_has_custom_game_id(self):
        """When include_game_id is a string, team_df should use that value."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id="custom_id_123", lazy=False)
        team = dataset.teams
        assert team["game_id"][0] == "custom_id_123"

    def test_string_player_df_has_custom_game_id(self):
        """When include_game_id is a string, player_df should use that value."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id="custom_id_123", lazy=False)
        player = dataset.players
        assert player["game_id"][0] == "custom_id_123"

    def test_string_metadata_df_has_custom_game_id(self):
        """When include_game_id is a string, metadata_df should also use that value."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id="custom_id_123", lazy=False)
        m = dataset.metadata
        assert m["game_id"][0] == "custom_id_123"

    # Test consistency
    def test_all_dataframes_have_same_game_id_when_true(self):
        """All DataFrames should have same game_id when include_game_id=True."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True, lazy=False)
        t = dataset.tracking
        m = dataset.metadata
        team = dataset.teams
        player = dataset.players
        game_id = m["game_id"][0]
        assert t["game_id"][0] == game_id
        assert team["game_id"][0] == game_id
        assert player["game_id"][0] == game_id


class TestIncludeGameIdSkillCorner:
    """Tests for include_game_id parameter with SkillCorner provider."""

    # Test include_game_id=True (default)
    def test_true_tracking_df_has_game_id(self):
        """When include_game_id=True, tracking_df should have game_id column."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=True, lazy=False)
        t = dataset.tracking
        assert "game_id" in t.columns

    def test_true_team_df_has_game_id(self):
        """When include_game_id=True, team_df should have game_id column."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=True, lazy=False)
        team = dataset.teams
        assert "game_id" in team.columns

    def test_true_player_df_has_game_id(self):
        """When include_game_id=True, player_df should have game_id column."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=True, lazy=False)
        player = dataset.players
        assert "game_id" in player.columns

    def test_true_metadata_df_has_game_id(self):
        """When include_game_id=True, metadata_df should have game_id column."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=True, lazy=False)
        m = dataset.metadata
        assert "game_id" in m.columns

    # Test include_game_id=False
    def test_false_tracking_df_no_game_id_column(self):
        """When include_game_id=False, tracking_df should not have game_id column."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=False, lazy=False)
        t = dataset.tracking
        assert "game_id" not in t.columns

    def test_false_team_df_no_game_id_column(self):
        """When include_game_id=False, team_df should not have game_id column."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=False, lazy=False)
        team = dataset.teams
        assert "game_id" not in team.columns

    def test_false_player_df_no_game_id_column(self):
        """When include_game_id=False, player_df should not have game_id column."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=False, lazy=False)
        player = dataset.players
        assert "game_id" not in player.columns

    def test_false_metadata_df_still_has_game_id(self):
        """When include_game_id=False, metadata_df should still have game_id column."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=False, lazy=False)
        m = dataset.metadata
        assert "game_id" in m.columns

    # Test include_game_id="custom_string"
    def test_string_tracking_df_has_custom_game_id(self):
        """When include_game_id is a string, tracking_df should use that value."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id="custom_sc_id", lazy=False)
        t = dataset.tracking
        assert t["game_id"][0] == "custom_sc_id"

    def test_string_team_df_has_custom_game_id(self):
        """When include_game_id is a string, team_df should use that value."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id="custom_sc_id", lazy=False)
        team = dataset.teams
        assert team["game_id"][0] == "custom_sc_id"

    def test_string_player_df_has_custom_game_id(self):
        """When include_game_id is a string, player_df should use that value."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id="custom_sc_id", lazy=False)
        player = dataset.players
        assert player["game_id"][0] == "custom_sc_id"

    def test_string_metadata_df_has_custom_game_id(self):
        """When include_game_id is a string, metadata_df should also use that value."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id="custom_sc_id", lazy=False)
        m = dataset.metadata
        assert m["game_id"][0] == "custom_sc_id"


class TestIncludeGameIdSportec:
    """Tests for include_game_id parameter with Sportec provider."""

    # Test include_game_id=True (default)
    def test_true_tracking_df_has_game_id(self):
        """When include_game_id=True, tracking_df should have game_id column."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id=True, lazy=False)
        t = dataset.tracking
        assert "game_id" in t.columns

    def test_true_team_df_has_game_id(self):
        """When include_game_id=True, team_df should have game_id column."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id=True, lazy=False)
        team = dataset.teams
        assert "game_id" in team.columns

    def test_true_player_df_has_game_id(self):
        """When include_game_id=True, player_df should have game_id column."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id=True, lazy=False)
        player = dataset.players
        assert "game_id" in player.columns

    def test_true_metadata_df_has_game_id(self):
        """When include_game_id=True, metadata_df should have game_id column."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id=True, lazy=False)
        m = dataset.metadata
        assert "game_id" in m.columns

    # Test include_game_id=False
    def test_false_tracking_df_no_game_id_column(self):
        """When include_game_id=False, tracking_df should not have game_id column."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id=False, lazy=False)
        t = dataset.tracking
        assert "game_id" not in t.columns

    def test_false_team_df_no_game_id_column(self):
        """When include_game_id=False, team_df should not have game_id column."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id=False, lazy=False)
        team = dataset.teams
        assert "game_id" not in team.columns

    def test_false_player_df_no_game_id_column(self):
        """When include_game_id=False, player_df should not have game_id column."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id=False, lazy=False)
        player = dataset.players
        assert "game_id" not in player.columns

    def test_false_metadata_df_still_has_game_id(self):
        """When include_game_id=False, metadata_df should still have game_id column."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id=False, lazy=False)
        m = dataset.metadata
        assert "game_id" in m.columns

    # Test include_game_id="custom_string"
    def test_string_tracking_df_has_custom_game_id(self):
        """When include_game_id is a string, tracking_df should use that value."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id="custom_sportec_id", lazy=False)
        t = dataset.tracking
        assert t["game_id"][0] == "custom_sportec_id"

    def test_string_team_df_has_custom_game_id(self):
        """When include_game_id is a string, team_df should use that value."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id="custom_sportec_id", lazy=False)
        team = dataset.teams
        assert team["game_id"][0] == "custom_sportec_id"

    def test_string_player_df_has_custom_game_id(self):
        """When include_game_id is a string, player_df should use that value."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id="custom_sportec_id", lazy=False)
        player = dataset.players
        assert player["game_id"][0] == "custom_sportec_id"

    def test_string_metadata_df_has_custom_game_id(self):
        """When include_game_id is a string, metadata_df should also use that value."""
        dataset = sportec.load_tracking(SP_RAW, SP_META, include_game_id="custom_sportec_id", lazy=False)
        m = dataset.metadata
        assert m["game_id"][0] == "custom_sportec_id"


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestIncludeGameIdLazy:
    """Tests for include_game_id with lazy loading."""

    def test_lazy_secondspectrum_string_game_id(self):
        """Lazy loading should also support custom game_id strings."""
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META, include_game_id="lazy_custom_id", lazy=True
        )
        t_lazy = dataset.tracking
        m = dataset.metadata
        t = dataset.tracking.collect()
        assert t["game_id"][0] == "lazy_custom_id"
        assert m["game_id"][0] == "lazy_custom_id"

    def test_lazy_skillcorner_string_game_id(self):
        """Lazy loading should also support custom game_id strings."""
        dataset = skillcorner.load_tracking(
            SC_RAW, SC_META, include_game_id="lazy_custom_id", lazy=True
        )
        t_lazy = dataset.tracking
        m = dataset.metadata
        t = dataset.tracking.collect()
        assert t["game_id"][0] == "lazy_custom_id"
        assert m["game_id"][0] == "lazy_custom_id"

    def test_lazy_sportec_string_game_id(self):
        """Lazy loading should also support custom game_id strings."""
        dataset = sportec.load_tracking(
            SP_RAW, SP_META, include_game_id="lazy_custom_id", lazy=True
        )
        t_lazy = dataset.tracking
        m = dataset.metadata
        t = dataset.tracking.collect()
        assert t["game_id"][0] == "lazy_custom_id"
        assert m["game_id"][0] == "lazy_custom_id"
