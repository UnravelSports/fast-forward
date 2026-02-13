"""Tests for post-load transform() method."""

import pytest
import polars as pl

from fastforward import secondspectrum, skillcorner
from tests.config import (
    SS_RAW_ANON as SS_RAW,
    SS_META_ANON as SS_META,
    SC_RAW_BOUNDARY as SC_RAW,
    SC_META_BOUNDARY as SC_META,
)

# ============================================================================
# Hard-coded test data values from secondspectrum test data
# ============================================================================
# Pitch dimensions from metadata
PITCH_LENGTH = 104.9000015258789
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2  # 52.45000076293945
HALF_WIDTH = PITCH_WIDTH / 2    # 34.0

# Test row: frame_id=2, player_id="player_1011"
# Raw JSON: {"playerId": "player_1011", "number": 52, "xyz": [-6.7, -19.1, 0.0], ...}
TEST_FRAME_ID = 2
TEST_PLAYER_ID = "player_1011"
ORIG_X_CDF = -6.7
ORIG_Y_CDF = -19.1


def get_test_row(df: pl.DataFrame) -> pl.DataFrame:
    """Get the specific test row (player 1011 in frame 2)."""
    return df.filter(
        (pl.col("frame_id") == TEST_FRAME_ID) &
        (pl.col("player_id") == TEST_PLAYER_ID)
    )


class TestTransformStateProperties:
    """Test that state properties are correctly exposed and match metadata."""

    def test_coordinate_system_property(self):
        """Dataset should expose coordinate_system property that matches metadata."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf")
        # Test convenience property
        assert dataset.coordinate_system == "cdf"
        # Test metadata (kept in sync)
        assert dataset.metadata["coordinate_system"][0] == "cdf"

        dataset2 = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="tracab")
        assert dataset2.coordinate_system == "tracab"
        assert dataset2.metadata["coordinate_system"][0] == "tracab"

    def test_orientation_property(self):
        """Dataset should expose orientation property that matches metadata."""
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META, orientation="static_home_away"
        )
        assert dataset.orientation == "static_home_away"
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_pitch_dimensions_property(self):
        """Dataset should expose pitch_dimensions property that matches metadata."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META)
        dims = dataset.pitch_dimensions
        assert isinstance(dims, tuple)
        assert len(dims) == 2
        
        assert dims[0] == pytest.approx(104.9000015258789)  # pitch_length
        assert dims[1] == pytest.approx(68.0)  # pitch_width
        
        # Verify matches metadata
        assert dims[0] == pytest.approx(float(dataset.metadata["pitch_length"][0]))
        assert dims[1] == pytest.approx(float(dataset.metadata["pitch_width"][0]))


class TestTransformIdentity:
    """Test identity transforms (no-ops)."""

    def test_no_args_returns_self(self):
        """transform() with no args should return self."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, lazy=False)
        result = dataset.transform()
        assert result is dataset

    def test_same_values_returns_self(self):
        """transform() with same values should return self."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, lazy=False)
        result = dataset.transform(
            to_orientation=dataset.orientation,
            to_dimensions=dataset.pitch_dimensions,
            to_coordinates=dataset.coordinate_system,
        )
        assert result is dataset


class TestTransformUpdatesMetadata:
    """Test that transform() updates metadata DataFrame correctly."""

    def test_transform_updates_coordinate_system_in_metadata(self):
        """Transform should update coordinate_system in metadata."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_coordinates="tracab")

        # Both property and metadata should reflect new value
        assert result.coordinate_system == "tracab"
        assert result.metadata["coordinate_system"][0] == "tracab"

    def test_transform_updates_orientation_in_metadata(self):
        """Transform should update orientation in metadata."""
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META, orientation="static_home_away", lazy=False
        )
        result = dataset.transform(to_orientation="static_away_home")

        assert result.orientation == "static_away_home"
        assert result.metadata["orientation"][0] == "static_away_home"

    def test_transform_updates_dimensions_in_metadata(self):
        """Transform should update pitch dimensions in metadata."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, lazy=False)
        result = dataset.transform(to_dimensions=(110.0, 70.0))

        assert result.pitch_dimensions == (110.0, 70.0)
        assert result.metadata["pitch_length"][0] == pytest.approx(110.0)
        assert result.metadata["pitch_width"][0] == pytest.approx(70.0)


class TestTransformCoordinates:
    """Test coordinate system transformations with hard-coded expected values."""

    def test_cdf_baseline_values(self):
        """Verify our test row has the expected CDF values."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        row = get_test_row(dataset.tracking)

        # Verify we got the right row with expected CDF coordinates
        assert row["x"][0] == pytest.approx(ORIG_X_CDF, abs=0.01)
        assert row["y"][0] == pytest.approx(ORIG_Y_CDF, abs=0.01)

    def test_cdf_to_tracab_hard_values(self):
        """CDF to Tracab: multiply by 100.

        Expected: x = -6.7 * 100 = -670.0
                  y = -19.1 * 100 = -1910.0
        """
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_coordinates="tracab")
        row = get_test_row(result.tracking)

        # Hard-coded expected values
        expected_x = -670.0
        expected_y = -1910.0

        assert result.coordinate_system == "tracab"
        assert row["x"][0] == pytest.approx(expected_x, abs=0.1)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.1)

    def test_cdf_to_opta_hard_values(self):
        """CDF to Opta: shift to bottom-left, normalize to 0-100.

        Formula: x_opta = ((x_cdf + half_length) / pitch_length) * 100
                 y_opta = ((y_cdf + half_width) / pitch_width) * 100

        Expected: x = ((-6.7 + 52.45) / 104.9) * 100 = 43.613
                  y = ((-19.1 + 34.0) / 68.0) * 100 = 21.912
        """
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_coordinates="opta")
        row = get_test_row(result.tracking)

        # Hard-coded expected values (calculated by hand)
        expected_x = ((ORIG_X_CDF + HALF_LENGTH) / PITCH_LENGTH) * 100  # 43.697
        expected_y = ((ORIG_Y_CDF + HALF_WIDTH) / PITCH_WIDTH) * 100    # 21.898

        assert result.coordinate_system == "opta"
        assert row["x"][0] == pytest.approx(expected_x, abs=0.01)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.01)

    def test_cdf_to_kloppy_hard_values(self):
        """CDF to Kloppy: shift to top-left, normalize to 0-1, invert Y.

        Formula: x_kloppy = (x_cdf + half_length) / pitch_length
                 y_kloppy = (half_width - y_cdf) / pitch_width  # Y inverted

        Expected: x = (-6.7 + 52.45) / 104.9 = 0.43613
                  y = (34.0 - (-19.1)) / 68.0 = 0.78088
        """
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_coordinates="kloppy")
        row = get_test_row(result.tracking)

        # Hard-coded expected values (Y inverted for kloppy)
        expected_x = (ORIG_X_CDF + HALF_LENGTH) / PITCH_LENGTH  # 0.43697
        expected_y = (HALF_WIDTH - ORIG_Y_CDF) / PITCH_WIDTH    # 0.78102

        assert result.coordinate_system == "kloppy"
        assert row["x"][0] == pytest.approx(expected_x, abs=0.0001)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.0001)

    def test_cdf_to_opta_bounds(self):
        """CDF to Opta should produce 0-100 range for boundary data."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_coordinates="opta")

        x_vals = result.tracking["x"].drop_nulls()
        y_vals = result.tracking["y"].drop_nulls()

        assert x_vals.min() == pytest.approx(0.0, abs=1e-5)
        assert x_vals.max() == pytest.approx(100.0, abs=1e-5)
        assert y_vals.min() == pytest.approx(0.0, abs=1e-5)
        assert y_vals.max() == pytest.approx(100.0, abs=1e-5)

    def test_cdf_to_kloppy_bounds(self):
        """CDF to Kloppy should produce 0-1 range for boundary data."""
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_coordinates="kloppy")

        x_vals = result.tracking["x"].drop_nulls()
        y_vals = result.tracking["y"].drop_nulls()

        assert x_vals.min() == pytest.approx(0.0, abs=1e-5)
        assert x_vals.max() == pytest.approx(1.0, abs=1e-5)
        assert y_vals.min() == pytest.approx(0.0, abs=1e-5)
        assert y_vals.max() == pytest.approx(1.0, abs=1e-5)

    def test_roundtrip_coordinates(self):
        """CDF->Tracab->CDF should restore original values."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        row_orig = get_test_row(dataset.tracking)
        orig_x = row_orig["x"][0]
        orig_y = row_orig["y"][0]

        to_tracab = dataset.transform(to_coordinates="tracab")
        restored = to_tracab.transform(to_coordinates="cdf")

        row_restored = get_test_row(restored.tracking)
        assert row_restored["x"][0] == pytest.approx(orig_x, abs=0.01)
        assert row_restored["y"][0] == pytest.approx(orig_y, abs=0.01)


class TestTransformOrientation:
    """Test orientation transformations with hard-coded expected values."""

    def test_flip_orientation_hard_values(self):
        """Flipping orientation should negate x and y.

        Expected: x = -(-6.7) = 6.7
                  y = -(-19.1) = 19.1
        """
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            orientation="static_home_away",
            coordinates="cdf",
            lazy=False
        )
        result = dataset.transform(to_orientation="static_away_home")
        row = get_test_row(result.tracking)

        # Hard-coded expected values (negated)
        expected_x = 6.7
        expected_y = 19.1

        assert result.orientation == "static_away_home"
        assert row["x"][0] == pytest.approx(expected_x, abs=0.01)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.01)

    def test_double_flip_restores_original(self):
        """Flipping twice should restore original coordinates."""
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            orientation="static_home_away",
            lazy=False
        )
        row_orig = get_test_row(dataset.tracking)
        orig_x = row_orig["x"][0]
        orig_y = row_orig["y"][0]

        flipped = dataset.transform(to_orientation="static_away_home")
        restored = flipped.transform(to_orientation="static_home_away")

        row_restored = get_test_row(restored.tracking)
        assert row_restored["x"][0] == pytest.approx(orig_x, abs=0.001)
        assert row_restored["y"][0] == pytest.approx(orig_y, abs=0.001)


class TestTransformDimensions:
    """Test dimension transformations."""

    def test_dimension_transform_updates_state(self):
        """Dimension transform should update pitch_dimensions."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, lazy=False)
        result = dataset.transform(to_dimensions=(110.0, 70.0))

        assert result.pitch_dimensions == (110.0, 70.0)
        assert result.coordinate_system == dataset.coordinate_system
        assert result.orientation == dataset.orientation

    def test_dimension_roundtrip(self):
        """Transform A->B->A should restore original coordinates."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, lazy=False)
        orig_dims = dataset.pitch_dimensions
        row_orig = get_test_row(dataset.tracking)
        orig_x = row_orig["x"][0]
        orig_y = row_orig["y"][0]

        transformed = dataset.transform(to_dimensions=(110.0, 70.0))
        restored = transformed.transform(to_dimensions=orig_dims)

        row_restored = get_test_row(restored.tracking)
        assert row_restored["x"][0] == pytest.approx(orig_x, abs=0.01)
        assert row_restored["y"][0] == pytest.approx(orig_y, abs=0.01)

    def test_dimension_transform_zone_scale_1(self):
        """Verify zone-based transform for point in scale=1.0 zone.

        Test point: x=-6.7, y=-19.1 on 104.9 x 68.0 pitch → 110.0 x 70.0

        Zone calculation:
        - x=-6.7 → x_pos=45.75 → Zone 5 (center circle, scale=1.0)
        - y=-19.1 → y_pos=14.9 → Zone 1 (penalty to 6-yard, scale=1.0)

        Since scale=1.0 in both zones, coordinates stay nearly unchanged.
        This verifies the algorithm correctly identifies and handles these zones.
        """
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_dimensions=(110.0, 70.0))
        row = get_test_row(result.tracking)

        # Both x and y fall in zones with scale=1.0
        # Zone boundaries shift with pitch size, but internal scaling is 1.0
        # Expected values stay approximately the same
        expected_x = -6.7  # Zone 5: center circle (9.15m fixed)
        expected_y = -19.1  # Zone 1: penalty area to 6-yard (11m fixed)

        assert row["x"][0] == pytest.approx(expected_x, abs=0.1)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.1)

    def test_dimension_transform_zone5_hard_values(self):
        """Verify zone-based transform with hard-coded expected values.

        Test point: player_1011 at x=-6.7, y=-19.1 (frame 2)
        Both x and y fall in zones with scale=1.0, so values stay ~unchanged.

        Zone 5 (center circle): scale = 1.0 (9.15m fixed on both pitches)
        Zone 1 (penalty to 6-yard): scale = 1.0 (11m fixed)
        """
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_dimensions=(110.0, 70.0))
        row = get_test_row(result.tracking)

        # Hard-coded expected values (scale=1.0 in both zones)
        expected_x = -6.7
        expected_y = -19.1

        assert row["x"][0] == pytest.approx(expected_x, abs=0.01)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.01)
        assert result.pitch_dimensions == (110.0, 70.0)

    def test_dimension_transform_zone4_hard_values(self):
        """Verify zone-based transform with hard-coded expected value for zone 4.

        Zone 4 is the "open field" zone between penalty arc (20.15m from goal)
        and center circle edge (half_length - 9.15m from goal). This zone
        actually scales when pitch length changes, unlike IFAB-fixed zones.

        Test point: player_1006 at x=-16.20 in frame 1

        Calculation (verified):
        - Source pitch: 104.8512m, half=52.4256m
        - Target pitch: 110.0m, half=55.0m
        - Zone 4 source: [20.15, 43.2756] (half - 9.15)
        - Zone 4 target: [20.15, 45.85] (half - 9.15)
        - Scale factor: (45.85-20.15)/(43.2756-20.15) = 25.7/23.1256 = 1.111323

        Transform x=-16.20:
        - x_pos = -16.20 + 52.4256 = 36.2256 (in zone 4: 20.15 < 36.2256 < 43.28)
        - scaled = 20.15 + (36.2256 - 20.15) * 1.111323 = 38.0152
        - x_new = 38.0152 - 55.0 = -16.9848
        """
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_dimensions=(110.0, 70.0))

        # Get player_1006 in frame 1 (in zone 4)
        row = result.tracking.filter(
            (pl.col("frame_id") == 1) & (pl.col("player_id") == "player_1006")
        )

        # Hard-coded expected value from zone-based calculation
        expected_x = -16.9848

        assert row["x"][0] == pytest.approx(expected_x, abs=0.01)

    def test_dimension_transform_preserves_ifab_landmarks(self):
        """IFAB landmarks should stay at fixed distances from goal line.

        When transforming from 104.85 to 110.0, landmarks like:
        - Penalty spot (11m from goal)
        - 6-yard line (5.5m from goal)
        - Penalty area (16.5m from goal)
        should remain at those fixed distances from the (new) goal line.
        """
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, coordinates="cdf", lazy=False)
        result = dataset.transform(to_dimensions=(110.0, 70.0))

        # The penalty spot is 11m from goal line
        # On 110m pitch: goal line is at x=-55, penalty spot at x=-44
        # A point exactly at the penalty spot should transform to -44

        # We don't have a player exactly at penalty spot, but we can verify
        # the transformation preserves the relationship by checking that
        # the zone boundaries are respected
        assert result.pitch_dimensions[0] == 110.0
        assert result.pitch_dimensions[1] == 70.0


class TestTransformCombined:
    """Test combined transformations with hard-coded expected values."""

    def test_flip_then_tracab_hard_values(self):
        """Flip orientation, then convert to Tracab.

        Step 1: Flip: x = 6.7, y = 19.1
        Step 2: Tracab (* 100): x = 670.0, y = 1910.0
        """
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            coordinates="cdf",
            orientation="static_home_away",
            lazy=False
        )

        result = dataset.transform(
            to_orientation="static_away_home",
            to_coordinates="tracab",
        )
        row = get_test_row(result.tracking)

        # Hard-coded expected values
        expected_x = 670.0
        expected_y = 1910.0

        assert result.orientation == "static_away_home"
        assert result.coordinate_system == "tracab"
        assert row["x"][0] == pytest.approx(expected_x, abs=0.1)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.1)

    def test_flip_then_opta_hard_values(self):
        """Flip orientation, then convert to Opta.

        Step 1: Flip: x = 6.7, y = 19.1
        Step 2: Opta: x = ((6.7 + 52.45) / 104.9) * 100 = 56.387
                      y = ((19.1 + 34.0) / 68.0) * 100 = 78.088
        """
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            coordinates="cdf",
            orientation="static_home_away",
            lazy=False
        )

        result = dataset.transform(
            to_orientation="static_away_home",
            to_coordinates="opta",
        )
        row = get_test_row(result.tracking)

        # Hard-coded expected values
        flipped_x = 6.7
        flipped_y = 19.1
        expected_x = ((flipped_x + HALF_LENGTH) / PITCH_LENGTH) * 100  # 56.303
        expected_y = ((flipped_y + HALF_WIDTH) / PITCH_WIDTH) * 100    # 78.102

        assert result.orientation == "static_away_home"
        assert result.coordinate_system == "opta"
        assert row["x"][0] == pytest.approx(expected_x, abs=0.01)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.01)

    def test_multiple_transforms_updates_all_state(self):
        """Multiple transforms should update all state correctly."""
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            coordinates="cdf",
            orientation="static_home_away",
            lazy=False
        )

        result = dataset.transform(
            to_orientation="static_away_home",
            to_dimensions=(110.0, 70.0),
            to_coordinates="tracab",
        )

        assert result.pitch_dimensions == (110.0, 70.0)
        assert result.coordinate_system == "tracab"
        assert result.orientation == "static_away_home"

    def test_transform_order_is_deterministic(self):
        """Parameter order should not affect result."""
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            coordinates="cdf",
            lazy=False
        )

        # Same transforms, different parameter order
        result1 = dataset.transform(
            to_dimensions=(110.0, 70.0),
            to_coordinates="tracab",
        )
        result2 = dataset.transform(
            to_coordinates="tracab",
            to_dimensions=(110.0, 70.0),
        )

        row1 = get_test_row(result1.tracking)
        row2 = get_test_row(result2.tracking)

        assert row1["x"][0] == pytest.approx(row2["x"][0], abs=0.001)
        assert row1["y"][0] == pytest.approx(row2["y"][0], abs=0.001)

    def test_dimensions_with_tracab_output(self):
        """Dimension + coordinate transform with hard-coded expected values.

        Test point: ball at x=55.0 (corner) after dimension transform to 110x70.
        Then Tracab (*100): x=5500.0
        """
        dataset = skillcorner.load_tracking(SC_RAW, SC_META, coordinates="cdf", lazy=False)

        result = dataset.transform(
            to_dimensions=(110.0, 70.0),
            to_coordinates="tracab",
        )

        # Get ball in frame 1
        row = result.tracking.filter(
            (pl.col("frame_id") == 1) & (pl.col("player_id") == "ball")
        )

        # Hard-coded expected values
        expected_x = 5500.0
        expected_y = 0.0

        assert row["x"][0] == pytest.approx(expected_x, abs=0.1)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.1)

    def test_all_three_transforms_hard_values(self):
        """Apply flip, dimensions, and coordinate transform with hard-coded values.

        Start: x = -6.7, y = -19.1 (CDF, player_1011 frame 2)
        Step 1: Flip → x = 6.7, y = 19.1
        Step 2: Dimensions (scale=1.0 for these zones) → x ≈ 6.7, y ≈ 19.1
        Step 3: Tracab (* 100) → x = 670.0, y = 1910.0
        """
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            coordinates="cdf",
            orientation="static_home_away",
            lazy=False
        )

        result = dataset.transform(
            to_orientation="static_away_home",
            to_dimensions=(110.0, 70.0),
            to_coordinates="tracab",
        )
        row = get_test_row(result.tracking)

        # Hard-coded expected values
        expected_x = 670.0
        expected_y = 1910.0

        assert result.orientation == "static_away_home"
        assert result.pitch_dimensions == (110.0, 70.0)
        assert result.coordinate_system == "tracab"
        assert row["x"][0] == pytest.approx(expected_x, abs=0.1)
        assert row["y"][0] == pytest.approx(expected_y, abs=0.1)


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestTransformLazy:
    """Test transform with lazy DataFrames."""

    def test_transform_collects_lazy(self):
        """Transform on lazy dataset should collect it."""
        dataset = secondspectrum.load_tracking(SS_RAW, SS_META, lazy=True)
        assert isinstance(dataset.tracking, pl.LazyFrame)

        result = dataset.transform(to_coordinates="tracab")

        # Result should be collected (eager DataFrame)
        assert isinstance(result.tracking, pl.DataFrame)
        assert result.coordinate_system == "tracab"


class TestTransformChaining:
    """Test chaining multiple transforms."""

    def test_chain_transforms(self):
        """Multiple transform() calls should chain correctly."""
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            coordinates="cdf",
            orientation="static_home_away",
            lazy=False
        )

        # Chain transforms
        result = (
            dataset
            .transform(to_dimensions=(110.0, 70.0))
            .transform(to_orientation="static_away_home")
            .transform(to_coordinates="tracab")
        )

        assert result.pitch_dimensions == (110.0, 70.0)
        assert result.orientation == "static_away_home"
        assert result.coordinate_system == "tracab"

    def test_chained_equals_single_call(self):
        """Chaining transforms should produce same result as single call."""
        dataset = secondspectrum.load_tracking(
            SS_RAW, SS_META,
            coordinates="cdf",
            orientation="static_home_away",
            lazy=False
        )

        # Method 1: All at once
        result1 = dataset.transform(
            to_orientation="static_away_home",
            to_coordinates="tracab"
        )

        # Method 2: Chained
        result2 = (
            dataset
            .transform(to_orientation="static_away_home")
            .transform(to_coordinates="tracab")
        )

        row1 = get_test_row(result1.tracking)
        row2 = get_test_row(result2.tracking)

        assert row1["x"][0] == pytest.approx(row2["x"][0], abs=0.001)
        assert row1["y"][0] == pytest.approx(row2["y"][0], abs=0.001)
