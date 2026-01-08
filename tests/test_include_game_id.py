"""Tests for include_game_id parameter across all providers."""

import pytest
import polars as pl
from pathlib import Path

from kloppy_light import secondspectrum, skillcorner, sportec

# Test data paths
DATA_DIR = Path(__file__).parent / "files"

# SecondSpectrum test files
SS_RAW = str(DATA_DIR / "secondspectrum_tracking.jsonl")
SS_META = str(DATA_DIR / "secondspectrum_meta.json")

# SkillCorner test files
SC_RAW = str(DATA_DIR / "skillcorner_tracking.jsonl")
SC_META = str(DATA_DIR / "skillcorner_meta.json")

# Sportec test files
SP_RAW = str(DATA_DIR / "sportec_positional.xml")
SP_META = str(DATA_DIR / "sportec_meta.xml")


class TestIncludeGameIdSecondSpectrum:
    """Tests for include_game_id parameter with SecondSpectrum provider."""

    # Test include_game_id=True (default)
    def test_true_tracking_df_has_game_id(self):
        """When include_game_id=True, tracking_df should have game_id column."""
        t, _, _, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True)
        assert "game_id" in t.columns
        assert t["game_id"][0] is not None

    def test_true_team_df_has_game_id(self):
        """When include_game_id=True, team_df should have game_id column."""
        _, _, team, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True)
        assert "game_id" in team.columns
        assert team["game_id"][0] is not None

    def test_true_player_df_has_game_id(self):
        """When include_game_id=True, player_df should have game_id column."""
        _, _, _, player = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True)
        assert "game_id" in player.columns
        assert player["game_id"][0] is not None

    def test_true_metadata_df_has_game_id(self):
        """When include_game_id=True, metadata_df should have game_id column."""
        _, m, _, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True)
        assert "game_id" in m.columns
        assert m["game_id"][0] is not None

    # Test include_game_id=False
    def test_false_tracking_df_no_game_id_column(self):
        """When include_game_id=False, tracking_df should not have game_id column."""
        t, _, _, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=False)
        assert "game_id" not in t.columns

    def test_false_team_df_no_game_id_column(self):
        """When include_game_id=False, team_df should not have game_id column."""
        _, _, team, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=False)
        assert "game_id" not in team.columns

    def test_false_player_df_no_game_id_column(self):
        """When include_game_id=False, player_df should not have game_id column."""
        _, _, _, player = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=False)
        assert "game_id" not in player.columns

    def test_false_metadata_df_still_has_game_id(self):
        """When include_game_id=False, metadata_df should still have game_id column."""
        _, m, _, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=False)
        assert "game_id" in m.columns
        assert m["game_id"][0] is not None

    # Test include_game_id="custom_string"
    def test_string_tracking_df_has_custom_game_id(self):
        """When include_game_id is a string, tracking_df should use that value."""
        t, _, _, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id="custom_id_123")
        assert t["game_id"][0] == "custom_id_123"

    def test_string_team_df_has_custom_game_id(self):
        """When include_game_id is a string, team_df should use that value."""
        _, _, team, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id="custom_id_123")
        assert team["game_id"][0] == "custom_id_123"

    def test_string_player_df_has_custom_game_id(self):
        """When include_game_id is a string, player_df should use that value."""
        _, _, _, player = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id="custom_id_123")
        assert player["game_id"][0] == "custom_id_123"

    def test_string_metadata_df_has_custom_game_id(self):
        """When include_game_id is a string, metadata_df should also use that value."""
        _, m, _, _ = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id="custom_id_123")
        assert m["game_id"][0] == "custom_id_123"

    # Test consistency
    def test_all_dataframes_have_same_game_id_when_true(self):
        """All DataFrames should have same game_id when include_game_id=True."""
        t, m, team, player = secondspectrum.load_tracking(SS_RAW, SS_META, include_game_id=True)
        game_id = m["game_id"][0]
        assert t["game_id"][0] == game_id
        assert team["game_id"][0] == game_id
        assert player["game_id"][0] == game_id


class TestIncludeGameIdSkillCorner:
    """Tests for include_game_id parameter with SkillCorner provider."""

    # Test include_game_id=True (default)
    def test_true_tracking_df_has_game_id(self):
        """When include_game_id=True, tracking_df should have game_id column."""
        t, _, _, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=True)
        assert "game_id" in t.columns

    def test_true_team_df_has_game_id(self):
        """When include_game_id=True, team_df should have game_id column."""
        _, _, team, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=True)
        assert "game_id" in team.columns

    def test_true_player_df_has_game_id(self):
        """When include_game_id=True, player_df should have game_id column."""
        _, _, _, player = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=True)
        assert "game_id" in player.columns

    def test_true_metadata_df_has_game_id(self):
        """When include_game_id=True, metadata_df should have game_id column."""
        _, m, _, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=True)
        assert "game_id" in m.columns

    # Test include_game_id=False
    def test_false_tracking_df_no_game_id_column(self):
        """When include_game_id=False, tracking_df should not have game_id column."""
        t, _, _, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=False)
        assert "game_id" not in t.columns

    def test_false_team_df_no_game_id_column(self):
        """When include_game_id=False, team_df should not have game_id column."""
        _, _, team, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=False)
        assert "game_id" not in team.columns

    def test_false_player_df_no_game_id_column(self):
        """When include_game_id=False, player_df should not have game_id column."""
        _, _, _, player = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=False)
        assert "game_id" not in player.columns

    def test_false_metadata_df_still_has_game_id(self):
        """When include_game_id=False, metadata_df should still have game_id column."""
        _, m, _, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id=False)
        assert "game_id" in m.columns

    # Test include_game_id="custom_string"
    def test_string_tracking_df_has_custom_game_id(self):
        """When include_game_id is a string, tracking_df should use that value."""
        t, _, _, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id="custom_sc_id")
        assert t["game_id"][0] == "custom_sc_id"

    def test_string_team_df_has_custom_game_id(self):
        """When include_game_id is a string, team_df should use that value."""
        _, _, team, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id="custom_sc_id")
        assert team["game_id"][0] == "custom_sc_id"

    def test_string_player_df_has_custom_game_id(self):
        """When include_game_id is a string, player_df should use that value."""
        _, _, _, player = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id="custom_sc_id")
        assert player["game_id"][0] == "custom_sc_id"

    def test_string_metadata_df_has_custom_game_id(self):
        """When include_game_id is a string, metadata_df should also use that value."""
        _, m, _, _ = skillcorner.load_tracking(SC_RAW, SC_META, include_game_id="custom_sc_id")
        assert m["game_id"][0] == "custom_sc_id"


class TestIncludeGameIdSportec:
    """Tests for include_game_id parameter with Sportec provider."""

    # Test include_game_id=True (default)
    def test_true_tracking_df_has_game_id(self):
        """When include_game_id=True, tracking_df should have game_id column."""
        t, _, _, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id=True)
        assert "game_id" in t.columns

    def test_true_team_df_has_game_id(self):
        """When include_game_id=True, team_df should have game_id column."""
        _, _, team, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id=True)
        assert "game_id" in team.columns

    def test_true_player_df_has_game_id(self):
        """When include_game_id=True, player_df should have game_id column."""
        _, _, _, player = sportec.load_tracking(SP_RAW, SP_META, include_game_id=True)
        assert "game_id" in player.columns

    def test_true_metadata_df_has_game_id(self):
        """When include_game_id=True, metadata_df should have game_id column."""
        _, m, _, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id=True)
        assert "game_id" in m.columns

    # Test include_game_id=False
    def test_false_tracking_df_no_game_id_column(self):
        """When include_game_id=False, tracking_df should not have game_id column."""
        t, _, _, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id=False)
        assert "game_id" not in t.columns

    def test_false_team_df_no_game_id_column(self):
        """When include_game_id=False, team_df should not have game_id column."""
        _, _, team, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id=False)
        assert "game_id" not in team.columns

    def test_false_player_df_no_game_id_column(self):
        """When include_game_id=False, player_df should not have game_id column."""
        _, _, _, player = sportec.load_tracking(SP_RAW, SP_META, include_game_id=False)
        assert "game_id" not in player.columns

    def test_false_metadata_df_still_has_game_id(self):
        """When include_game_id=False, metadata_df should still have game_id column."""
        _, m, _, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id=False)
        assert "game_id" in m.columns

    # Test include_game_id="custom_string"
    def test_string_tracking_df_has_custom_game_id(self):
        """When include_game_id is a string, tracking_df should use that value."""
        t, _, _, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id="custom_sportec_id")
        assert t["game_id"][0] == "custom_sportec_id"

    def test_string_team_df_has_custom_game_id(self):
        """When include_game_id is a string, team_df should use that value."""
        _, _, team, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id="custom_sportec_id")
        assert team["game_id"][0] == "custom_sportec_id"

    def test_string_player_df_has_custom_game_id(self):
        """When include_game_id is a string, player_df should use that value."""
        _, _, _, player = sportec.load_tracking(SP_RAW, SP_META, include_game_id="custom_sportec_id")
        assert player["game_id"][0] == "custom_sportec_id"

    def test_string_metadata_df_has_custom_game_id(self):
        """When include_game_id is a string, metadata_df should also use that value."""
        _, m, _, _ = sportec.load_tracking(SP_RAW, SP_META, include_game_id="custom_sportec_id")
        assert m["game_id"][0] == "custom_sportec_id"


class TestIncludeGameIdLazy:
    """Tests for include_game_id with lazy loading."""

    def test_lazy_secondspectrum_string_game_id(self):
        """Lazy loading should also support custom game_id strings."""
        t_lazy, m, _, _ = secondspectrum.load_tracking(
            SS_RAW, SS_META, include_game_id="lazy_custom_id", lazy=True
        )
        t = t_lazy.collect()
        assert t["game_id"][0] == "lazy_custom_id"
        assert m["game_id"][0] == "lazy_custom_id"

    def test_lazy_skillcorner_string_game_id(self):
        """Lazy loading should also support custom game_id strings."""
        t_lazy, m, _, _ = skillcorner.load_tracking(
            SC_RAW, SC_META, include_game_id="lazy_custom_id", lazy=True
        )
        t = t_lazy.collect()
        assert t["game_id"][0] == "lazy_custom_id"
        assert m["game_id"][0] == "lazy_custom_id"

    def test_lazy_sportec_string_game_id(self):
        """Lazy loading should also support custom game_id strings."""
        t_lazy, m, _, _ = sportec.load_tracking(
            SP_RAW, SP_META, include_game_id="lazy_custom_id", lazy=True
        )
        t = t_lazy.collect()
        assert t["game_id"][0] == "lazy_custom_id"
        assert m["game_id"][0] == "lazy_custom_id"
