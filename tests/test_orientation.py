"""Tests for orientation transformations across periods."""

import pytest
import polars as pl
from pathlib import Path

from kloppy_light import secondspectrum

# Test data paths
DATA_DIR = Path(__file__).parent / "files"
RAW_DATA_PATH = str(DATA_DIR / "secondspectrum_tracking.jsonl")
META_DATA_PATH = str(DATA_DIR / "secondspectrum_meta.json")


class TestStaticHomeAway:
    """Tests for static_home_away orientation."""

    def test_home_team_x_period_1(self):
        """Home team should have x mean of -13.010109 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_home_away", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(-13.010109067540617, rel=1e-6)

    def test_home_team_x_period_2(self):
        """Home team should have x mean of -8.234799 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_home_away", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(-8.234799977654422, rel=1e-6)

    def test_ball_x_period_1(self):
        """Ball should have x mean of 20.0939 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_home_away", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(20.093899972736835, rel=1e-6)

    def test_ball_x_period_2(self):
        """Ball should have x mean of -6.3284 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_home_away", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(-6.3283999802544715, rel=1e-6)


class TestStaticAwayHome:
    """Tests for static_away_home orientation."""

    def test_home_team_x_period_1(self):
        """Home team should have x mean of 13.010109 in period 1 (flipped from static_home_away)."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(13.010109067540617, rel=1e-6)

    def test_home_team_x_period_2(self):
        """Home team should have x mean of 8.234799 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(8.234799977654422, rel=1e-6)

    def test_ball_x_period_1(self):
        """Ball should have x mean of -20.0939 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(-20.093899972736835, rel=1e-6)

    def test_ball_x_period_2(self):
        """Ball should have x mean of 6.3284 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(6.3283999802544715, rel=1e-6)


class TestHomeAway:
    """Tests for home_away orientation (alternating)."""

    def test_home_team_x_period_1(self):
        """Home team should have x mean of -13.010109 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="home_away", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(-13.010109067540617, rel=1e-6)

    def test_home_team_x_period_2(self):
        """Home team should have x mean of 8.234799 in period 2 (flipped)."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="home_away", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(8.234799977654422, rel=1e-6)

    def test_ball_x_period_1(self):
        """Ball should have x mean of 20.0939 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="home_away", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(20.093899972736835, rel=1e-6)

    def test_ball_x_period_2(self):
        """Ball should have x mean of 6.3284 in period 2 (flipped)."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="home_away", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(6.3283999802544715, rel=1e-6)


class TestAwayHome:
    """Tests for away_home orientation (alternating)."""

    def test_home_team_x_period_1(self):
        """Home team should have x mean of 13.010109 in period 1 (flipped)."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="away_home", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(13.010109067540617, rel=1e-6)

    def test_home_team_x_period_2(self):
        """Home team should have x mean of -8.234799 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="away_home", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(-8.234799977654422, rel=1e-6)

    def test_ball_x_period_1(self):
        """Ball should have x mean of -20.0939 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="away_home", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(-20.093899972736835, rel=1e-6)

    def test_ball_x_period_2(self):
        """Ball should have x mean of -6.3284 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="away_home", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(-6.3283999802544715, rel=1e-6)


class TestAttackRight:
    """Tests for attack_right orientation."""

    def test_home_team_x_period_1(self):
        """Home team should have x mean of 12.738454 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_right", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(12.738454520466991, rel=1e-6)

    def test_home_team_x_period_2(self):
        """Home team should have x mean of -8.234799 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_right", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(-8.234799977654422, rel=1e-6)

    def test_ball_x_period_1(self):
        """Ball should have x mean of -20.0883 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_right", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(-20.088299972712992, rel=1e-6)

    def test_ball_x_period_2(self):
        """Ball should have x mean of -6.3284 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_right", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(-6.3283999802544715, rel=1e-6)


class TestAttackLeft:
    """Tests for attack_left orientation."""

    def test_home_team_x_period_1(self):
        """Home team should have x mean of -12.738454 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_left", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(-12.738454520466991, rel=1e-6)

    def test_home_team_x_period_2(self):
        """Home team should have x mean of 8.234799 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_left", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking
        team_df = dataset.teams

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
        period_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == home_team_id)
        )
        avg_x = period_data["x"].mean()
        assert avg_x == pytest.approx(8.234799977654422, rel=1e-6)

    def test_ball_x_period_1(self):
        """Ball should have x mean of 20.0883 in period 1."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_left", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(20.088299972712992, rel=1e-6)

    def test_ball_x_period_2(self):
        """Ball should have x mean of 6.3284 in period 2."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="attack_left", only_alive=False, lazy=False
        )
        tracking_df = dataset.tracking

        ball_data = tracking_df.filter(
            (pl.col("period_id") == 2) & (pl.col("team_id") == "ball")
        )
        avg_x = ball_data["x"].mean()
        assert avg_x == pytest.approx(6.3283999802544715, rel=1e-6)


class TestOrientationSymmetry:
    """Tests for symmetry between orientation modes."""

    def test_static_modes_are_symmetric(self):
        """static_home_away and static_away_home should produce symmetric results."""
        dataset_home = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_home_away", only_alive=False, lazy=False
        )
        tracking_home = dataset_home.tracking
        team_df = dataset_home.teams
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, orientation="static_away_home", only_alive=False, lazy=False
        )
        tracking_away = dataset.tracking

        home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]

        # Get home team positions in both modes
        home_in_home_mode = tracking_home.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == home_team_id)
        )
        home_in_away_mode = tracking_away.filter(
            (pl.col("period_id") == 1) & (pl.col("team_id") == home_team_id)
        )

        # X values should be negated
        assert home_in_home_mode["x"].mean() == pytest.approx(-home_in_away_mode["x"].mean(), rel=1e-6)
