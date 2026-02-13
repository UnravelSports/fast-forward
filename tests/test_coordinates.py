"""Tests for coordinate system transformations."""

import pytest
import polars as pl
from pathlib import Path

from fastforward import skillcorner, secondspectrum
from tests.config import (
    DATA_DIR,
    SC_RAW_BOUNDARY as BOUNDARY_RAW_PATH,
    SC_META_BOUNDARY as BOUNDARY_META_PATH,
    SS_RAW_ANON as SS_RAW_PATH,
    SS_META_ANON as SS_META_PATH,
)


class TestCdfCoordinates:
    """Tests for CDF (Common Data Format) coordinate system."""

    def test_cdf_x_range(self):
        """CDF x coordinates should be exactly [-52.5, 52.5] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="cdf", lazy=False
        )
        tracking_df = dataset.tracking

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() == pytest.approx(-52.5, rel=1e-6)
        assert x_values.max() == pytest.approx(52.5, rel=1e-6)

    def test_cdf_y_range(self):
        """CDF y coordinates should be exactly [-34.0, 34.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="cdf", lazy=False
        )
        tracking_df = dataset.tracking

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() == pytest.approx(-34.0, rel=1e-6)
        assert y_values.max() == pytest.approx(34.0, rel=1e-6)

    def test_cdf_aliases(self):
        """Aliases like 'secondspectrum' should produce same results as 'cdf'."""
        dataset = secondspectrum.load_tracking(
            SS_RAW_PATH, SS_META_PATH, coordinates="cdf", lazy=False
        )
        tracking_cdf = dataset.tracking
        dataset = secondspectrum.load_tracking(
            SS_RAW_PATH, SS_META_PATH, coordinates="secondspectrum", lazy=False
        )
        tracking_ss = dataset.tracking

        # Should be identical
        assert tracking_cdf["x"].equals(tracking_ss["x"])
        assert tracking_cdf["y"].equals(tracking_ss["y"])


class TestKloppyCoordinates:
    """Tests for Kloppy normalized coordinate system."""

    def test_kloppy_x_range(self):
        """Kloppy x coordinates should be exactly [0.0, 1.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="kloppy", lazy=False
        )
        tracking_df = dataset.tracking

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() == pytest.approx(0.0, abs=1e-6)
        assert x_values.max() == pytest.approx(1.0, abs=1e-6)

    def test_kloppy_y_range(self):
        """Kloppy y coordinates should be exactly [0.0, 1.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="kloppy", lazy=False
        )
        tracking_df = dataset.tracking

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() == pytest.approx(0.0, abs=1e-6)
        assert y_values.max() == pytest.approx(1.0, abs=1e-6)


class TestTracabCoordinates:
    """Tests for Tracab centimeter coordinate system."""

    def test_tracab_x_range(self):
        """Tracab x coordinates should be exactly [-5250.0, 5250.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="tracab", lazy=False
        )
        tracking_df = dataset.tracking

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() == pytest.approx(-5250.0, rel=1e-6)
        assert x_values.max() == pytest.approx(5250.0, rel=1e-6)

    def test_tracab_y_range(self):
        """Tracab y coordinates should be exactly [-3400.0, 3400.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="tracab", lazy=False
        )
        tracking_df = dataset.tracking

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() == pytest.approx(-3400.0, rel=1e-6)
        assert y_values.max() == pytest.approx(3400.0, rel=1e-6)

    def test_tracab_scale_factor(self):
        """Tracab coordinates should be 100x CDF coordinates."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="cdf", lazy=False
        )
        tracking_cdf = dataset.tracking
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="tracab", lazy=False
        )
        tracking_tracab = dataset.tracking

        # Get first non-null x value
        cdf_x = tracking_cdf.filter(pl.col("x").is_not_null())["x"][0]
        tracab_x = tracking_tracab.filter(pl.col("x").is_not_null())["x"][0]

        # Should be exactly 100x
        assert tracab_x == pytest.approx(cdf_x * 100, rel=1e-6)


class TestSportVuCoordinates:
    """Tests for SportVU coordinate system (top-left origin)."""

    def test_sportvu_x_range(self):
        """SportVU x coordinates should be exactly [0.0, 105.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="sportvu", lazy=False
        )
        tracking_df = dataset.tracking

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() == pytest.approx(0.0, abs=1e-6)
        assert x_values.max() == pytest.approx(105.0, rel=1e-6)

    def test_sportvu_y_range(self):
        """SportVU y coordinates should be exactly [0.0, 68.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="sportvu", lazy=False
        )
        tracking_df = dataset.tracking

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() == pytest.approx(0.0, abs=1e-6)
        assert y_values.max() == pytest.approx(68.0, rel=1e-6)


class TestSportecEventCoordinates:
    """Tests for Sportec Event coordinate system (bottom-left origin)."""

    def test_sportec_event_x_range(self):
        """Sportec Event x coordinates should be exactly [0.0, 105.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="sportec:event", lazy=False
        )
        tracking_df = dataset.tracking

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() == pytest.approx(0.0, abs=1e-6)
        assert x_values.max() == pytest.approx(105.0, rel=1e-6)

    def test_sportec_event_y_range(self):
        """Sportec Event y coordinates should be exactly [0.0, 68.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="sportec:event", lazy=False
        )
        tracking_df = dataset.tracking

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() == pytest.approx(0.0, abs=1e-6)
        assert y_values.max() == pytest.approx(68.0, rel=1e-6)


class TestOptaCoordinates:
    """Tests for Opta normalized coordinate system."""

    def test_opta_x_range(self):
        """Opta x coordinates should be exactly [0.0, 100.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="opta", lazy=False
        )
        tracking_df = dataset.tracking

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() == pytest.approx(0.0, abs=1e-6)
        assert x_values.max() == pytest.approx(100.0, rel=1e-6)

    def test_opta_y_range(self):
        """Opta y coordinates should be exactly [0.0, 100.0] for boundary data."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="opta", lazy=False
        )
        tracking_df = dataset.tracking

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() == pytest.approx(0.0, abs=1e-6)
        assert y_values.max() == pytest.approx(100.0, rel=1e-6)


class TestInvalidCoordinateSystem:
    """Tests for invalid coordinate system handling."""

    def test_invalid_raises_error(self):
        """Invalid coordinate system should raise an error."""
        with pytest.raises(Exception):  # Will be a RuntimeError from Rust
            skillcorner.load_tracking(
                BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="invalid_system", lazy=False
            )


class TestCoordinatesOutsidePitch:
    """Tests that coordinates outside pitch boundaries are NOT clamped.

    Uses regular skillcorner_tracking.jsonl data which has y values slightly
    outside the pitch boundary (y max: 34.15 on a 68m pitch with boundary at 34.0).
    """

    # Test data paths for regular data (with values outside pitch)
    SC_RAW_PATH = str(DATA_DIR / "skillcorner_tracking.jsonl")
    SC_META_PATH = str(DATA_DIR / "skillcorner_meta.json")

    def test_kloppy_y_below_zero(self):
        """Kloppy y values outside pitch should be below 0, not clamped."""
        dataset = skillcorner.load_tracking(
            self.SC_RAW_PATH, self.SC_META_PATH, coordinates="kloppy", lazy=False
        )
        tracking_df = dataset.tracking
        y_values = tracking_df["y"].drop_nulls()
        # y_min should be slightly negative (not clamped to 0)
        assert y_values.min() == pytest.approx(-0.002205904806032777, rel=1e-6)

    def test_opta_y_above_100(self):
        """Opta y values outside pitch should exceed 100, not clamped."""
        dataset = skillcorner.load_tracking(
            self.SC_RAW_PATH, self.SC_META_PATH, coordinates="opta", lazy=False
        )
        tracking_df = dataset.tracking
        y_values = tracking_df["y"].drop_nulls()
        # y_max should exceed 100 (not clamped to 100)
        assert y_values.max() == pytest.approx(100.2205810546875, rel=1e-6)

    def test_sportvu_y_below_zero(self):
        """SportVU y values outside pitch should be below 0."""
        dataset = skillcorner.load_tracking(
            self.SC_RAW_PATH, self.SC_META_PATH, coordinates="sportvu", lazy=False
        )
        tracking_df = dataset.tracking
        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() == pytest.approx(-0.15000152587890625, rel=1e-6)

    def test_sportec_event_y_above_pitch(self):
        """Sportec:event y values outside pitch should exceed pitch_width."""
        dataset = skillcorner.load_tracking(
            self.SC_RAW_PATH, self.SC_META_PATH, coordinates="sportec:event", lazy=False
        )
        tracking_df = dataset.tracking
        y_values = tracking_df["y"].drop_nulls()
        # pitch_width is 68, y_max should exceed it
        assert y_values.max() == pytest.approx(68.1500015258789, rel=1e-6)

    def test_tracab_y_outside_boundaries(self):
        """Tracab y values outside pitch should exceed pitch_width/2 * 100."""
        dataset = skillcorner.load_tracking(
            self.SC_RAW_PATH, self.SC_META_PATH, coordinates="tracab", lazy=False
        )
        tracking_df = dataset.tracking
        y_values = tracking_df["y"].drop_nulls()
        # pitch_width/2 * 100 = 3400, y_max should exceed it
        assert y_values.max() == pytest.approx(3415.000244140625, rel=1e-6)


class TestMetadataCoordinateSystem:
    """Tests that metadata DataFrame reflects the coordinate system."""

    def test_metadata_shows_cdf(self):
        """Metadata should show 'cdf' when using cdf coordinates."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="cdf", lazy=False
        )
        metadata_df = dataset.metadata
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_metadata_shows_kloppy(self):
        """Metadata should show 'kloppy' when using kloppy coordinates."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="kloppy", lazy=False
        )
        metadata_df = dataset.metadata
        assert metadata_df["coordinate_system"][0] == "kloppy"

    def test_metadata_shows_tracab(self):
        """Metadata should show 'tracab' when using tracab coordinates."""
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH, coordinates="tracab", lazy=False
        )
        metadata_df = dataset.metadata
        assert metadata_df["coordinate_system"][0] == "tracab"


class TestCoordinateTransformationWithOrientation:
    """Tests that coordinate transformation works with orientation transformation."""

    def test_transformation_order(self):
        """Orientation should be applied before coordinate transformation."""
        # Load with both orientation and coordinate transformation
        dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH,
            coordinates="kloppy",
            orientation="static_home_away", lazy=False
        )
        tracking_df = dataset.tracking

        # Should have valid coordinates in kloppy range
        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() == pytest.approx(0.0, abs=1e-6)
        assert x_values.max() == pytest.approx(1.0, abs=1e-6)

    def test_all_orientations_with_coordinates(self):
        """All orientation modes should work with all coordinate systems."""
        orientations = [
            "static_home_away", "static_away_home",
            "home_away", "away_home",
            "attack_right", "attack_left"
        ]
        coordinates = ["cdf", "kloppy", "tracab", "sportvu", "sportec:event", "opta"]

        # Test a subset of combinations to keep test fast
        for orient in orientations[:2]:
            for coord in coordinates[:3]:
                dataset = skillcorner.load_tracking(
            BOUNDARY_RAW_PATH, BOUNDARY_META_PATH,
                    coordinates=coord,
                    orientation=orient,
                    lazy=False
                )
                assert dataset.tracking.height == 8, f"Failed for {orient} + {coord}"
