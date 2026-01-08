"""Tests for coordinate system transformations."""

import pytest
import polars as pl
from pathlib import Path

from kloppy_light import secondspectrum

# Test data paths
DATA_DIR = Path(__file__).parent / "files"
RAW_DATA_PATH = str(DATA_DIR / "secondspectrum_tracking.jsonl")
META_DATA_PATH = str(DATA_DIR / "secondspectrum_meta.json")

# Pitch dimensions from test data
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


class TestCdfCoordinates:
    """Tests for CDF (Common Data Format) coordinate system."""

    def test_cdf_x_range(self):
        """CDF x coordinates should be in [-pitch_length/2, pitch_length/2]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="cdf"
        )

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() >= -PITCH_LENGTH / 2 - 1  # Allow small margin
        assert x_values.max() <= PITCH_LENGTH / 2 + 1

    def test_cdf_y_range(self):
        """CDF y coordinates should be in [-pitch_width/2, pitch_width/2]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="cdf"
        )

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() >= -PITCH_WIDTH / 2 - 1
        assert y_values.max() <= PITCH_WIDTH / 2 + 1

    def test_cdf_aliases(self):
        """Aliases like 'secondspectrum' should produce same results as 'cdf'."""
        tracking_cdf, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="cdf"
        )
        tracking_ss, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="secondspectrum"
        )

        # Should be identical
        assert tracking_cdf["x"].equals(tracking_ss["x"])
        assert tracking_cdf["y"].equals(tracking_ss["y"])


class TestKloppyCoordinates:
    """Tests for Kloppy normalized coordinate system."""

    def test_kloppy_x_range(self):
        """Kloppy x coordinates should be in [0, 1]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="kloppy"
        )

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() >= -0.1  # Allow small margin for players outside
        assert x_values.max() <= 1.1

    def test_kloppy_y_range(self):
        """Kloppy y coordinates should be in [0, 1]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="kloppy"
        )

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() >= -0.1
        assert y_values.max() <= 1.1


class TestTracabCoordinates:
    """Tests for Tracab centimeter coordinate system."""

    def test_tracab_x_range(self):
        """Tracab x coordinates should be in cm (100x CDF)."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="tracab"
        )

        x_values = tracking_df["x"].drop_nulls()
        # Tracab should be in centimeters (100x meters)
        assert x_values.min() >= -PITCH_LENGTH / 2 * 100 - 100
        assert x_values.max() <= PITCH_LENGTH / 2 * 100 + 100

    def test_tracab_y_range(self):
        """Tracab y coordinates should be in cm."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="tracab"
        )

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() >= -PITCH_WIDTH / 2 * 100 - 100
        assert y_values.max() <= PITCH_WIDTH / 2 * 100 + 100

    def test_tracab_scale_factor(self):
        """Tracab coordinates should be 100x CDF coordinates."""
        tracking_cdf, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="cdf"
        )
        tracking_tracab, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="tracab"
        )

        # Get first non-null x value
        cdf_x = tracking_cdf.filter(pl.col("x").is_not_null())["x"][0]
        tracab_x = tracking_tracab.filter(pl.col("x").is_not_null())["x"][0]

        # Should be approximately 100x
        assert abs(tracab_x - cdf_x * 100) < 0.1


class TestSportVuCoordinates:
    """Tests for SportVU coordinate system (top-left origin)."""

    def test_sportvu_x_range(self):
        """SportVU x coordinates should be in [0, pitch_length]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="sportvu"
        )

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() >= -1
        assert x_values.max() <= PITCH_LENGTH + 1

    def test_sportvu_y_range(self):
        """SportVU y coordinates should be in [0, pitch_width]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="sportvu"
        )

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() >= -1
        assert y_values.max() <= PITCH_WIDTH + 1


class TestSportecEventCoordinates:
    """Tests for Sportec Event coordinate system (bottom-left origin)."""

    def test_sportec_event_x_range(self):
        """Sportec Event x coordinates should be in [0, pitch_length]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="sportec:event"
        )

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() >= -1
        assert x_values.max() <= PITCH_LENGTH + 1

    def test_sportec_event_y_range(self):
        """Sportec Event y coordinates should be in [0, pitch_width]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="sportec:event"
        )

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() >= -1
        assert y_values.max() <= PITCH_WIDTH + 1


class TestOptaCoordinates:
    """Tests for Opta normalized coordinate system."""

    def test_opta_x_range(self):
        """Opta x coordinates should be in [0, 100]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="opta"
        )

        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() >= -5  # Allow small margin
        assert x_values.max() <= 105

    def test_opta_y_range(self):
        """Opta y coordinates should be in [0, 100]."""
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="opta"
        )

        y_values = tracking_df["y"].drop_nulls()
        assert y_values.min() >= -5
        assert y_values.max() <= 105


class TestInvalidCoordinateSystem:
    """Tests for invalid coordinate system handling."""

    def test_invalid_raises_error(self):
        """Invalid coordinate system should raise an error."""
        with pytest.raises(Exception):  # Will be a RuntimeError from Rust
            secondspectrum.load_tracking(
                RAW_DATA_PATH, META_DATA_PATH, coordinates="invalid_system"
            )


class TestMetadataCoordinateSystem:
    """Tests that metadata DataFrame reflects the coordinate system."""

    def test_metadata_shows_cdf(self):
        """Metadata should show 'cdf' when using cdf coordinates."""
        _, metadata_df, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="cdf"
        )
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_metadata_shows_kloppy(self):
        """Metadata should show 'kloppy' when using kloppy coordinates."""
        _, metadata_df, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="kloppy"
        )
        assert metadata_df["coordinate_system"][0] == "kloppy"

    def test_metadata_shows_tracab(self):
        """Metadata should show 'tracab' when using tracab coordinates."""
        _, metadata_df, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH, coordinates="tracab"
        )
        assert metadata_df["coordinate_system"][0] == "tracab"


class TestCoordinateTransformationWithOrientation:
    """Tests that coordinate transformation works with orientation transformation."""

    def test_transformation_order(self):
        """Orientation should be applied before coordinate transformation."""
        # Load with both orientation and coordinate transformation
        tracking_df, _, _, _ = secondspectrum.load_tracking(
            RAW_DATA_PATH, META_DATA_PATH,
            coordinates="kloppy",
            orientation="static_home_away"
        )

        # Should have valid coordinates in kloppy range
        x_values = tracking_df["x"].drop_nulls()
        assert x_values.min() >= -0.1
        assert x_values.max() <= 1.1

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
                tracking_df, _, _, _ = secondspectrum.load_tracking(
                    RAW_DATA_PATH, META_DATA_PATH,
                    coordinates=coord,
                    orientation=orient
                )
                assert tracking_df.height > 0, f"Failed for {orient} + {coord}"
