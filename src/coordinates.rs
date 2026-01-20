use crate::error::KloppyError;

/// Coordinate system for tracking data
///
/// All coordinate systems can be transformed to/from CDF (Common Data Format),
/// which is used as the intermediate format for bidirectional transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateSystem {
    /// Common Data Format: origin at center, meters
    /// X ∈ [-pitch_length/2, +pitch_length/2] (left to right)
    /// Y ∈ [-pitch_width/2, +pitch_width/2] (bottom to top)
    Cdf,

    /// Kloppy normalized: origin at top-left, normalized 0-1
    /// X ∈ [0, 1] (left to right)
    /// Y ∈ [0, 1] (top to bottom, inverted from CDF)
    Kloppy,

    /// Tracab: origin at center, centimeters
    /// X ∈ [-pitch_length/2 * 100, +pitch_length/2 * 100] (left to right)
    /// Y ∈ [-pitch_width/2 * 100, +pitch_width/2 * 100] (bottom to top)
    Tracab,

    /// SportVU: origin at top-left, meters
    /// X ∈ [0, pitch_length] (left to right)
    /// Y ∈ [0, pitch_width] (top to bottom, inverted from CDF)
    SportVu,

    /// Sportec Event: origin at bottom-left, meters
    /// X ∈ [0, pitch_length] (left to right)
    /// Y ∈ [0, pitch_width] (bottom to top)
    SportecEvent,

    /// Opta: origin at bottom-left, normalized 0-100
    /// X ∈ [0, 100] (left to right)
    /// Y ∈ [0, 100] (bottom to top)
    Opta,
}

impl CoordinateSystem {
    /// Parse coordinate system from string.
    /// Multiple names may map to the same coordinate system (aliases).
    pub fn from_str(s: &str) -> Result<Self, KloppyError> {
        match s.to_lowercase().as_str() {
            // CDF and aliases
            "cdf" | "secondspectrum" | "skillcorner" | "pff" | "gradientsports" | "sportec:tracking" | "hawkeye" | "signality" => {
                Ok(CoordinateSystem::Cdf)
            }
            // Distinct coordinate systems
            "kloppy" => Ok(CoordinateSystem::Kloppy),
            "tracab" => Ok(CoordinateSystem::Tracab),
            "sportvu" | "statsperform" => Ok(CoordinateSystem::SportVu),
            "sportec:event" | "respovision" => Ok(CoordinateSystem::SportecEvent),
            "opta" => Ok(CoordinateSystem::Opta),
            _ => Err(KloppyError::UnsupportedCoordinates(s.to_string())),
        }
    }

    /// Get the string representation of this coordinate system
    pub fn as_str(&self) -> &'static str {
        match self {
            CoordinateSystem::Cdf => "cdf",
            CoordinateSystem::Kloppy => "kloppy",
            CoordinateSystem::Tracab => "tracab",
            CoordinateSystem::SportVu => "sportvu",
            CoordinateSystem::SportecEvent => "sportec:event",
            CoordinateSystem::Opta => "opta",
        }
    }
}

/// Transform coordinates from CDF to target coordinate system.
///
/// CDF (Common Data Format) is used as the base/intermediate coordinate system.
/// All transformations go through CDF: source → CDF → target
///
/// # Arguments
/// * `x`, `y`, `z` - Coordinates in CDF (center origin, meters)
/// * `target` - Target coordinate system
/// * `pitch_length` - Pitch length in meters
/// * `pitch_width` - Pitch width in meters
///
/// # Returns
/// Transformed (x, y, z) coordinates in the target system
pub fn transform_from_cdf(
    x: f32,
    y: f32,
    z: f32,
    target: CoordinateSystem,
    pitch_length: f32,
    pitch_width: f32,
) -> (f32, f32, f32) {
    match target {
        CoordinateSystem::Cdf => (x, y, z),

        CoordinateSystem::Kloppy => {
            // CDF → Kloppy: shift origin to top-left, normalize to 0-1, invert Y
            let new_x = (x + pitch_length / 2.0) / pitch_length;
            let new_y = (pitch_width / 2.0 - y) / pitch_width; // Y inverted
            (new_x, new_y, z)
        }

        CoordinateSystem::Tracab => {
            // CDF → Tracab: convert meters to centimeters
            (x * 100.0, y * 100.0, z * 100.0)
        }

        CoordinateSystem::SportVu => {
            // CDF → SportVU: shift origin to top-left, invert Y
            let new_x = x + pitch_length / 2.0;
            let new_y = pitch_width / 2.0 - y; // Y inverted
            (new_x, new_y, z)
        }

        CoordinateSystem::SportecEvent => {
            // CDF → Sportec Event: shift origin to bottom-left
            let new_x = x + pitch_length / 2.0;
            let new_y = y + pitch_width / 2.0;
            (new_x, new_y, z)
        }

        CoordinateSystem::Opta => {
            // CDF → Opta: shift origin to bottom-left, normalize to 0-100
            let new_x = ((x + pitch_length / 2.0) / pitch_length) * 100.0;
            let new_y = ((y + pitch_width / 2.0) / pitch_width) * 100.0;
            (new_x, new_y, z)
        }
    }
}

/// Transform coordinates from source coordinate system to CDF.
///
/// # Arguments
/// * `x`, `y`, `z` - Coordinates in source system
/// * `source` - Source coordinate system
/// * `pitch_length` - Pitch length in meters
/// * `pitch_width` - Pitch width in meters
///
/// # Returns
/// Transformed (x, y, z) coordinates in CDF
pub fn transform_to_cdf(
    x: f32,
    y: f32,
    z: f32,
    source: CoordinateSystem,
    pitch_length: f32,
    pitch_width: f32,
) -> (f32, f32, f32) {
    match source {
        CoordinateSystem::Cdf => (x, y, z),

        CoordinateSystem::Kloppy => {
            // Kloppy → CDF: denormalize from 0-1, shift origin to center, uninvert Y
            let new_x = x * pitch_length - pitch_length / 2.0;
            let new_y = pitch_width / 2.0 - y * pitch_width; // Y uninverted
            (new_x, new_y, z)
        }

        CoordinateSystem::Tracab => {
            // Tracab → CDF: convert centimeters to meters
            (x / 100.0, y / 100.0, z / 100.0)
        }

        CoordinateSystem::SportVu => {
            // SportVU → CDF: shift origin to center, uninvert Y
            let new_x = x - pitch_length / 2.0;
            let new_y = pitch_width / 2.0 - y; // Y uninverted
            (new_x, new_y, z)
        }

        CoordinateSystem::SportecEvent => {
            // Sportec Event → CDF: shift origin to center
            let new_x = x - pitch_length / 2.0;
            let new_y = y - pitch_width / 2.0;
            (new_x, new_y, z)
        }

        CoordinateSystem::Opta => {
            // Opta → CDF: denormalize from 0-100, shift origin to center
            let new_x = (x / 100.0) * pitch_length - pitch_length / 2.0;
            let new_y = (y / 100.0) * pitch_width - pitch_width / 2.0;
            (new_x, new_y, z)
        }
    }
}

// =============================================================================
// Unit enum for coordinate system units
// =============================================================================

/// Unit of measurement for coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Meters,
    Centimeters,
    Yards,
    Feet,
    /// Normalized coordinates (0-1 or 0-100 scale)
    Normed,
}

impl Unit {
    /// Get the unit for a coordinate system
    pub fn for_coordinate_system(cs: CoordinateSystem) -> Self {
        match cs {
            CoordinateSystem::Cdf => Unit::Meters,
            CoordinateSystem::Kloppy => Unit::Normed,
            CoordinateSystem::Tracab => Unit::Centimeters,
            CoordinateSystem::SportVu => Unit::Meters,
            CoordinateSystem::SportecEvent => Unit::Meters,
            CoordinateSystem::Opta => Unit::Normed,
        }
    }

    /// Conversion factor to meters (for non-normed units)
    pub fn to_meters_factor(&self) -> f32 {
        match self {
            Unit::Meters => 1.0,
            Unit::Centimeters => 0.01,
            Unit::Yards => 0.9144,
            Unit::Feet => 0.3048,
            Unit::Normed => 1.0, // Requires pitch dimensions for actual conversion
        }
    }
}

// =============================================================================
// Zone-based dimension transformation (IFAB standard)
// =============================================================================

// IFAB standard pitch markings (in meters) - compile-time constants
const IFAB_GOAL_WIDTH: f32 = 7.32;
const IFAB_SIX_YARD_LENGTH: f32 = 5.5;
const IFAB_SIX_YARD_WIDTH: f32 = 18.32;
const IFAB_PENALTY_AREA_LENGTH: f32 = 16.5;
const IFAB_PENALTY_AREA_WIDTH: f32 = 40.32;
const IFAB_PENALTY_SPOT: f32 = 11.0;
const IFAB_PENALTY_ARC: f32 = 9.15;
const IFAB_CENTER_CIRCLE: f32 = 9.15;

const NUM_X_ZONES: usize = 6;
const NUM_Y_ZONES: usize = 4;

/// Pre-computed transformation context for fast zone-based dimension transforms.
///
/// This struct enables zone-based non-linear scaling of coordinates to preserve
/// pitch feature proportions (penalty area, six-yard box, center circle, etc.)
/// when transforming between different pitch dimensions.
///
/// The transformation divides the pitch into zones based on IFAB standard markings:
/// - X-axis: 6 zones (goal line, 6-yard, penalty spot, penalty area, penalty arc, center)
/// - Y-axis: 4 zones (sideline, penalty area edge, 6-yard edge, goalpost)
///
/// Zone boundaries (like the penalty spot at 11m) remain at fixed distances from
/// the goal line, while zones between them scale proportionally.
#[derive(Clone, Copy)]
pub struct DimensionTransform {
    pub from_length: f32,
    pub from_width: f32,
    pub to_length: f32,
    pub to_width: f32,
    // Pre-computed zone boundaries and scale factors
    x_from_bounds: [f32; NUM_X_ZONES + 1],
    x_to_bounds: [f32; NUM_X_ZONES + 1],
    x_scales: [f32; NUM_X_ZONES],
    y_from_bounds: [f32; NUM_Y_ZONES + 1],
    y_to_bounds: [f32; NUM_Y_ZONES + 1],
    y_scales: [f32; NUM_Y_ZONES],
}

impl DimensionTransform {
    /// Create a new transformation context.
    ///
    /// Call this once and reuse for all coordinate transformations.
    /// Zone boundaries and scale factors are pre-computed for performance.
    pub fn new(from_length: f32, from_width: f32, to_length: f32, to_width: f32) -> Self {
        let penalty_arc = IFAB_PENALTY_SPOT + IFAB_PENALTY_ARC;

        // X zone boundaries (from goal line = 0 to center = half_length)
        let from_half_x = from_length / 2.0;
        let to_half_x = to_length / 2.0;

        let x_from_bounds = [
            0.0,
            IFAB_SIX_YARD_LENGTH,
            IFAB_PENALTY_SPOT,
            IFAB_PENALTY_AREA_LENGTH,
            penalty_arc,
            from_half_x - IFAB_CENTER_CIRCLE,
            from_half_x,
        ];

        let x_to_bounds = [
            0.0,
            IFAB_SIX_YARD_LENGTH,
            IFAB_PENALTY_SPOT,
            IFAB_PENALTY_AREA_LENGTH,
            penalty_arc,
            to_half_x - IFAB_CENTER_CIRCLE,
            to_half_x,
        ];

        // Pre-compute scale factors for each zone
        let mut x_scales = [1.0f32; NUM_X_ZONES];
        for i in 0..NUM_X_ZONES {
            let from_size = x_from_bounds[i + 1] - x_from_bounds[i];
            let to_size = x_to_bounds[i + 1] - x_to_bounds[i];
            x_scales[i] = if from_size > 0.0 {
                to_size / from_size
            } else {
                1.0
            };
        }

        // Y zone boundaries (from sideline = 0 to center = half_width)
        let from_half_y = from_width / 2.0;
        let to_half_y = to_width / 2.0;

        let y_from_bounds = [
            0.0,
            (from_width - IFAB_PENALTY_AREA_WIDTH) / 2.0,
            (from_width - IFAB_SIX_YARD_WIDTH) / 2.0,
            (from_width - IFAB_GOAL_WIDTH) / 2.0,
            from_half_y,
        ];

        let y_to_bounds = [
            0.0,
            (to_width - IFAB_PENALTY_AREA_WIDTH) / 2.0,
            (to_width - IFAB_SIX_YARD_WIDTH) / 2.0,
            (to_width - IFAB_GOAL_WIDTH) / 2.0,
            to_half_y,
        ];

        let mut y_scales = [1.0f32; NUM_Y_ZONES];
        for i in 0..NUM_Y_ZONES {
            let from_size = y_from_bounds[i + 1] - y_from_bounds[i];
            let to_size = y_to_bounds[i + 1] - y_to_bounds[i];
            y_scales[i] = if from_size > 0.0 {
                to_size / from_size
            } else {
                1.0
            };
        }

        Self {
            from_length,
            from_width,
            to_length,
            to_width,
            x_from_bounds,
            x_to_bounds,
            x_scales,
            y_from_bounds,
            y_to_bounds,
            y_scales,
        }
    }

    /// Check if transformation is identity (no change needed)
    #[inline(always)]
    pub fn is_identity(&self) -> bool {
        (self.from_length - self.to_length).abs() < 0.001
            && (self.from_width - self.to_width).abs() < 0.001
    }

    /// Transform a single X coordinate (CDF: center origin)
    #[inline(always)]
    pub fn transform_x(&self, x: f32) -> f32 {
        // Convert to positive origin (goal line = 0)
        let x_pos = x + self.from_length / 2.0;

        // Handle second half by mirroring
        let (v, mirror) = if x_pos > self.x_from_bounds[NUM_X_ZONES] {
            (self.from_length - x_pos, true)
        } else {
            (x_pos, false)
        };

        // Find zone using linear search (only 6 zones, faster than binary for small N)
        let mut result = v;
        for i in 0..NUM_X_ZONES {
            if v <= self.x_from_bounds[i + 1] {
                result = self.x_to_bounds[i] + (v - self.x_from_bounds[i]) * self.x_scales[i];
                break;
            }
        }

        // Mirror back and convert to CDF origin
        let result = if mirror {
            self.to_length - result
        } else {
            result
        };
        result - self.to_length / 2.0
    }

    /// Transform a single Y coordinate (CDF: center origin)
    #[inline(always)]
    pub fn transform_y(&self, y: f32) -> f32 {
        // Convert to positive origin (sideline = 0)
        let y_pos = y + self.from_width / 2.0;

        // Handle second half by mirroring
        let (v, mirror) = if y_pos > self.y_from_bounds[NUM_Y_ZONES] {
            (self.from_width - y_pos, true)
        } else {
            (y_pos, false)
        };

        // Find zone
        let mut result = v;
        for i in 0..NUM_Y_ZONES {
            if v <= self.y_from_bounds[i + 1] {
                result = self.y_to_bounds[i] + (v - self.y_from_bounds[i]) * self.y_scales[i];
                break;
            }
        }

        // Mirror back and convert to CDF origin
        let result = if mirror {
            self.to_width - result
        } else {
            result
        };
        result - self.to_width / 2.0
    }

    /// Transform x, y, z coordinates (z passes through unchanged)
    #[inline(always)]
    pub fn transform(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        (self.transform_x(x), self.transform_y(y), z)
    }
}

/// Convenience function for one-off dimension transformations.
///
/// For repeated transformations, create a `DimensionTransform` once and reuse it.
pub fn transform_pitch_dimensions(
    x: f32,
    y: f32,
    z: f32,
    from_length: f32,
    from_width: f32,
    to_length: f32,
    to_width: f32,
) -> (f32, f32, f32) {
    let transform = DimensionTransform::new(from_length, from_width, to_length, to_width);
    if transform.is_identity() {
        (x, y, z)
    } else {
        transform.transform(x, y, z)
    }
}

// =============================================================================
// Coordinate system transformations
// =============================================================================

/// Transform coordinates between any two coordinate systems.
///
/// Uses CDF as intermediate format: source → CDF → target
///
/// # Arguments
/// * `x`, `y`, `z` - Coordinates in source system
/// * `source` - Source coordinate system
/// * `target` - Target coordinate system
/// * `pitch_length` - Pitch length in meters
/// * `pitch_width` - Pitch width in meters
///
/// # Returns
/// Transformed (x, y, z) coordinates in target system
pub fn transform_coordinates(
    x: f32,
    y: f32,
    z: f32,
    source: CoordinateSystem,
    target: CoordinateSystem,
    pitch_length: f32,
    pitch_width: f32,
) -> (f32, f32, f32) {
    if source == target {
        return (x, y, z);
    }

    // Transform via CDF: source → CDF → target
    let (cdf_x, cdf_y, cdf_z) = transform_to_cdf(x, y, z, source, pitch_length, pitch_width);
    transform_from_cdf(cdf_x, cdf_y, cdf_z, target, pitch_length, pitch_width)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PITCH_LENGTH: f32 = 105.0;
    const PITCH_WIDTH: f32 = 68.0;

    // ========================================================================
    // CoordinateSystem::from_str tests
    // ========================================================================

    #[test]
    fn test_from_str_cdf() {
        assert_eq!(CoordinateSystem::from_str("cdf").unwrap(), CoordinateSystem::Cdf);
        assert_eq!(CoordinateSystem::from_str("CDF").unwrap(), CoordinateSystem::Cdf);
    }

    #[test]
    fn test_from_str_cdf_aliases() {
        // All these should map to CDF
        assert_eq!(CoordinateSystem::from_str("secondspectrum").unwrap(), CoordinateSystem::Cdf);
        assert_eq!(CoordinateSystem::from_str("skillcorner").unwrap(), CoordinateSystem::Cdf);
        assert_eq!(CoordinateSystem::from_str("pff").unwrap(), CoordinateSystem::Cdf);
        assert_eq!(CoordinateSystem::from_str("sportec:tracking").unwrap(), CoordinateSystem::Cdf);
        assert_eq!(CoordinateSystem::from_str("hawkeye").unwrap(), CoordinateSystem::Cdf);
    }

    #[test]
    fn test_from_str_distinct_systems() {
        assert_eq!(CoordinateSystem::from_str("kloppy").unwrap(), CoordinateSystem::Kloppy);
        assert_eq!(CoordinateSystem::from_str("tracab").unwrap(), CoordinateSystem::Tracab);
        assert_eq!(CoordinateSystem::from_str("sportvu").unwrap(), CoordinateSystem::SportVu);
        assert_eq!(CoordinateSystem::from_str("sportec:event").unwrap(), CoordinateSystem::SportecEvent);
        assert_eq!(CoordinateSystem::from_str("opta").unwrap(), CoordinateSystem::Opta);
    }

    #[test]
    fn test_from_str_invalid() {
        assert!(CoordinateSystem::from_str("invalid").is_err());
    }

    // ========================================================================
    // CDF → Target transformation tests
    // ========================================================================

    #[test]
    fn test_cdf_to_cdf() {
        let (x, y, z) = transform_from_cdf(10.0, 20.0, 1.5, CoordinateSystem::Cdf, PITCH_LENGTH, PITCH_WIDTH);
        assert_eq!(x, 10.0);
        assert_eq!(y, 20.0);
        assert_eq!(z, 1.5);
    }

    #[test]
    fn test_cdf_to_kloppy_center() {
        // CDF center (0, 0) → Kloppy center (0.5, 0.5)
        let (x, y, _) = transform_from_cdf(0.0, 0.0, 0.0, CoordinateSystem::Kloppy, PITCH_LENGTH, PITCH_WIDTH);
        assert!((x - 0.5).abs() < 0.001);
        assert!((y - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cdf_to_kloppy_corners() {
        // CDF top-left (-52.5, 34) → Kloppy top-left (0, 0)
        let (x, y, _) = transform_from_cdf(-PITCH_LENGTH/2.0, PITCH_WIDTH/2.0, 0.0, CoordinateSystem::Kloppy, PITCH_LENGTH, PITCH_WIDTH);
        assert!(x.abs() < 0.001);
        assert!(y.abs() < 0.001);

        // CDF bottom-right (52.5, -34) → Kloppy bottom-right (1, 1)
        let (x, y, _) = transform_from_cdf(PITCH_LENGTH/2.0, -PITCH_WIDTH/2.0, 0.0, CoordinateSystem::Kloppy, PITCH_LENGTH, PITCH_WIDTH);
        assert!((x - 1.0).abs() < 0.001);
        assert!((y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cdf_to_tracab() {
        // CDF meters → Tracab centimeters
        let (x, y, z) = transform_from_cdf(10.0, 20.0, 1.5, CoordinateSystem::Tracab, PITCH_LENGTH, PITCH_WIDTH);
        assert_eq!(x, 1000.0);
        assert_eq!(y, 2000.0);
        assert_eq!(z, 150.0);
    }

    #[test]
    fn test_cdf_to_sportvu_center() {
        // CDF center (0, 0) → SportVU (pitch_length/2, pitch_width/2)
        let (x, y, _) = transform_from_cdf(0.0, 0.0, 0.0, CoordinateSystem::SportVu, PITCH_LENGTH, PITCH_WIDTH);
        assert!((x - PITCH_LENGTH/2.0).abs() < 0.001);
        assert!((y - PITCH_WIDTH/2.0).abs() < 0.001);
    }

    #[test]
    fn test_cdf_to_sportec_event_center() {
        // CDF center (0, 0) → Sportec Event (pitch_length/2, pitch_width/2)
        let (x, y, _) = transform_from_cdf(0.0, 0.0, 0.0, CoordinateSystem::SportecEvent, PITCH_LENGTH, PITCH_WIDTH);
        assert!((x - PITCH_LENGTH/2.0).abs() < 0.001);
        assert!((y - PITCH_WIDTH/2.0).abs() < 0.001);
    }

    #[test]
    fn test_cdf_to_opta_center() {
        // CDF center (0, 0) → Opta center (50, 50)
        let (x, y, _) = transform_from_cdf(0.0, 0.0, 0.0, CoordinateSystem::Opta, PITCH_LENGTH, PITCH_WIDTH);
        assert!((x - 50.0).abs() < 0.001);
        assert!((y - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_cdf_to_opta_corners() {
        // CDF bottom-left (-52.5, -34) → Opta bottom-left (0, 0)
        let (x, y, _) = transform_from_cdf(-PITCH_LENGTH/2.0, -PITCH_WIDTH/2.0, 0.0, CoordinateSystem::Opta, PITCH_LENGTH, PITCH_WIDTH);
        assert!(x.abs() < 0.001);
        assert!(y.abs() < 0.001);

        // CDF top-right (52.5, 34) → Opta top-right (100, 100)
        let (x, y, _) = transform_from_cdf(PITCH_LENGTH/2.0, PITCH_WIDTH/2.0, 0.0, CoordinateSystem::Opta, PITCH_LENGTH, PITCH_WIDTH);
        assert!((x - 100.0).abs() < 0.001);
        assert!((y - 100.0).abs() < 0.001);
    }

    // ========================================================================
    // Round-trip transformation tests (CDF → Target → CDF)
    // ========================================================================

    fn assert_round_trip(target: CoordinateSystem, x: f32, y: f32, z: f32) {
        let (tx, ty, tz) = transform_from_cdf(x, y, z, target, PITCH_LENGTH, PITCH_WIDTH);
        let (rx, ry, rz) = transform_to_cdf(tx, ty, tz, target, PITCH_LENGTH, PITCH_WIDTH);
        assert!((rx - x).abs() < 0.001, "X mismatch for {:?}: {} vs {}", target, rx, x);
        assert!((ry - y).abs() < 0.001, "Y mismatch for {:?}: {} vs {}", target, ry, y);
        assert!((rz - z).abs() < 0.001, "Z mismatch for {:?}: {} vs {}", target, rz, z);
    }

    #[test]
    fn test_round_trip_kloppy() {
        assert_round_trip(CoordinateSystem::Kloppy, 0.0, 0.0, 0.0);
        assert_round_trip(CoordinateSystem::Kloppy, 10.0, 20.0, 1.5);
        assert_round_trip(CoordinateSystem::Kloppy, -30.0, -15.0, 0.0);
    }

    #[test]
    fn test_round_trip_tracab() {
        assert_round_trip(CoordinateSystem::Tracab, 0.0, 0.0, 0.0);
        assert_round_trip(CoordinateSystem::Tracab, 10.0, 20.0, 1.5);
        assert_round_trip(CoordinateSystem::Tracab, -30.0, -15.0, 0.0);
    }

    #[test]
    fn test_round_trip_sportvu() {
        assert_round_trip(CoordinateSystem::SportVu, 0.0, 0.0, 0.0);
        assert_round_trip(CoordinateSystem::SportVu, 10.0, 20.0, 1.5);
        assert_round_trip(CoordinateSystem::SportVu, -30.0, -15.0, 0.0);
    }

    #[test]
    fn test_round_trip_sportec_event() {
        assert_round_trip(CoordinateSystem::SportecEvent, 0.0, 0.0, 0.0);
        assert_round_trip(CoordinateSystem::SportecEvent, 10.0, 20.0, 1.5);
        assert_round_trip(CoordinateSystem::SportecEvent, -30.0, -15.0, 0.0);
    }

    #[test]
    fn test_round_trip_opta() {
        assert_round_trip(CoordinateSystem::Opta, 0.0, 0.0, 0.0);
        assert_round_trip(CoordinateSystem::Opta, 10.0, 20.0, 1.5);
        assert_round_trip(CoordinateSystem::Opta, -30.0, -15.0, 0.0);
    }

    // ========================================================================
    // Bidirectional transformation tests (any → any)
    // ========================================================================

    #[test]
    fn test_transform_same_system() {
        let (x, y, z) = transform_coordinates(10.0, 20.0, 1.5, CoordinateSystem::Kloppy, CoordinateSystem::Kloppy, PITCH_LENGTH, PITCH_WIDTH);
        assert_eq!(x, 10.0);
        assert_eq!(y, 20.0);
        assert_eq!(z, 1.5);
    }

    #[test]
    fn test_transform_kloppy_to_opta() {
        // Kloppy (0.5, 0.5) → Opta (50, 50)
        let (x, y, _) = transform_coordinates(0.5, 0.5, 0.0, CoordinateSystem::Kloppy, CoordinateSystem::Opta, PITCH_LENGTH, PITCH_WIDTH);
        assert!((x - 50.0).abs() < 0.001);
        assert!((y - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_transform_tracab_to_sportvu() {
        // Tracab center (0, 0) → SportVU (pitch_length/2, pitch_width/2)
        let (x, y, _) = transform_coordinates(0.0, 0.0, 0.0, CoordinateSystem::Tracab, CoordinateSystem::SportVu, PITCH_LENGTH, PITCH_WIDTH);
        assert!((x - PITCH_LENGTH/2.0).abs() < 0.001);
        assert!((y - PITCH_WIDTH/2.0).abs() < 0.001);
    }

    // ========================================================================
    // DimensionTransform tests (zone-based pitch dimension transformation)
    // ========================================================================

    // Zone boundaries (from goal line, in meters)
    const SIX_YARD: f32 = 5.5;
    const PENALTY_SPOT: f32 = 11.0;
    const PENALTY_AREA: f32 = 16.5;
    const PENALTY_ARC: f32 = 20.15; // 11.0 + 9.15
    const CENTER_CIRCLE: f32 = 9.15;

    #[test]
    fn test_dimension_transform_identity() {
        let transform = DimensionTransform::new(105.0, 68.0, 105.0, 68.0);
        assert!(transform.is_identity());

        // Any point should remain unchanged
        assert!((transform.transform_x(0.0) - 0.0).abs() < 0.001);
        assert!((transform.transform_x(25.0) - 25.0).abs() < 0.001);
        assert!((transform.transform_x(-52.5) - (-52.5)).abs() < 0.001);
        assert!((transform.transform_y(0.0) - 0.0).abs() < 0.001);
        assert!((transform.transform_y(20.0) - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_dimension_transform_zone_boundaries_x() {
        // Transform from 110x70 to 100x66
        let transform = DimensionTransform::new(110.0, 70.0, 100.0, 66.0);

        // Goal line: from -55 to -50
        let from_goal = -55.0;
        let to_goal = -50.0;
        assert!(
            (transform.transform_x(from_goal) - to_goal).abs() < 0.001,
            "Goal line: expected {}, got {}",
            to_goal,
            transform.transform_x(from_goal)
        );

        // 6-yard line: from_goal + 5.5m should map to to_goal + 5.5m
        let from_6yard = from_goal + SIX_YARD;
        let to_6yard = to_goal + SIX_YARD;
        assert!(
            (transform.transform_x(from_6yard) - to_6yard).abs() < 0.001,
            "6-yard line: expected {}, got {}",
            to_6yard,
            transform.transform_x(from_6yard)
        );

        // Penalty spot: from_goal + 11m should map to to_goal + 11m
        let from_penalty_spot = from_goal + PENALTY_SPOT;
        let to_penalty_spot = to_goal + PENALTY_SPOT;
        assert!(
            (transform.transform_x(from_penalty_spot) - to_penalty_spot).abs() < 0.001,
            "Penalty spot: expected {}, got {}",
            to_penalty_spot,
            transform.transform_x(from_penalty_spot)
        );

        // Penalty area: from_goal + 16.5m should map to to_goal + 16.5m
        let from_penalty_area = from_goal + PENALTY_AREA;
        let to_penalty_area = to_goal + PENALTY_AREA;
        assert!(
            (transform.transform_x(from_penalty_area) - to_penalty_area).abs() < 0.001,
            "Penalty area: expected {}, got {}",
            to_penalty_area,
            transform.transform_x(from_penalty_area)
        );

        // Center (x=0) should stay at x=0
        assert!(
            (transform.transform_x(0.0) - 0.0).abs() < 0.001,
            "Center: expected 0, got {}",
            transform.transform_x(0.0)
        );
    }

    #[test]
    fn test_dimension_transform_zone_boundaries_y() {
        // Transform from 110x70 to 100x66
        let transform = DimensionTransform::new(110.0, 70.0, 100.0, 66.0);

        // Goal post positions: center ± 3.66 (half of 7.32m goal width)
        // These should remain fixed
        assert!(
            (transform.transform_y(3.66) - 3.66).abs() < 0.001,
            "Goal post +: expected 3.66, got {}",
            transform.transform_y(3.66)
        );
        assert!(
            (transform.transform_y(-3.66) - (-3.66)).abs() < 0.001,
            "Goal post -: expected -3.66, got {}",
            transform.transform_y(-3.66)
        );

        // Center (y=0) should stay at y=0
        assert!(
            (transform.transform_y(0.0) - 0.0).abs() < 0.001,
            "Center: expected 0, got {}",
            transform.transform_y(0.0)
        );
    }

    #[test]
    fn test_dimension_transform_zone_scaling() {
        // From 110x70 to 100x66
        let transform = DimensionTransform::new(110.0, 70.0, 100.0, 66.0);

        // Point at x=30 (in the "open field" zone between penalty arc and center circle)
        // Zone 4: from [20.15, 55-9.15=45.85], to [20.15, 50-9.15=40.85]
        // Scale factor: (40.85-20.15)/(45.85-20.15) = 20.7/25.7 ≈ 0.8054
        // x_out = 20.15 + (30 - 20.15) * 0.8054 ≈ 28.08
        let x_in = 30.0;
        let x_out = transform.transform_x(x_in);
        assert!(
            (x_out - 28.08).abs() < 0.2,
            "Zone scaling: expected ~28.08, got {}",
            x_out
        );
    }

    #[test]
    fn test_dimension_transform_mirror_symmetry() {
        let transform = DimensionTransform::new(110.0, 70.0, 100.0, 66.0);

        // Points symmetric around center should transform symmetrically
        let x_left = -30.0;
        let x_right = 30.0;
        let out_left = transform.transform_x(x_left);
        let out_right = transform.transform_x(x_right);

        assert!(
            (out_left - (-out_right)).abs() < 0.001,
            "X symmetry: {} should equal -{}, diff = {}",
            out_left,
            out_right,
            (out_left + out_right).abs()
        );

        // Same for Y axis
        let y_top = 20.0;
        let y_bottom = -20.0;
        let out_top = transform.transform_y(y_top);
        let out_bottom = transform.transform_y(y_bottom);

        assert!(
            (out_top - (-out_bottom)).abs() < 0.001,
            "Y symmetry: {} should equal -{}, diff = {}",
            out_top,
            out_bottom,
            (out_top + out_bottom).abs()
        );
    }

    #[test]
    fn test_dimension_transform_boundary_corners() {
        let transform = DimensionTransform::new(105.0, 68.0, 110.0, 70.0);

        // Corner: (-52.5, -34) -> should map to (-55, -35)
        let x = transform.transform_x(-52.5);
        let y = transform.transform_y(-34.0);
        assert!(
            (x - (-55.0)).abs() < 0.001,
            "Corner X: expected -55, got {}",
            x
        );
        assert!(
            (y - (-35.0)).abs() < 0.001,
            "Corner Y: expected -35, got {}",
            y
        );

        // Opposite corner
        let x2 = transform.transform_x(52.5);
        let y2 = transform.transform_y(34.0);
        assert!(
            (x2 - 55.0).abs() < 0.001,
            "Corner X opposite: expected 55, got {}",
            x2
        );
        assert!(
            (y2 - 35.0).abs() < 0.001,
            "Corner Y opposite: expected 35, got {}",
            y2
        );
    }

    #[test]
    fn test_dimension_transform_penalty_area_width_preserved() {
        // Only change width, keep length same
        let transform = DimensionTransform::new(105.0, 68.0, 105.0, 72.0);

        // Penalty area width is 40.32m, centered at y=0
        // So edges are at y = ±20.16
        // These should remain at ±20.16 regardless of pitch width
        assert!(
            (transform.transform_y(20.16) - 20.16).abs() < 0.001,
            "Penalty area edge +: expected 20.16, got {}",
            transform.transform_y(20.16)
        );
        assert!(
            (transform.transform_y(-20.16) - (-20.16)).abs() < 0.001,
            "Penalty area edge -: expected -20.16, got {}",
            transform.transform_y(-20.16)
        );

        // Point outside penalty area should scale outward with wider pitch
        let y_out = transform.transform_y(30.0);
        assert!(
            y_out > 30.0,
            "Point outside penalty area should scale outward: {} should be > 30",
            y_out
        );
    }

    #[test]
    fn test_dimension_transform_ifab_landmarks() {
        // Transform from IFAB standard to larger pitch
        let transform = DimensionTransform::new(105.0, 68.0, 110.0, 70.0);

        // Penalty spot: 11m from goal line = -52.5 + 11 = -41.5
        // Should map to: -55 + 11 = -44 on 110m pitch
        let penalty_spot_x = -41.5;
        assert!(
            (transform.transform_x(penalty_spot_x) - (-44.0)).abs() < 0.001,
            "Penalty spot: expected -44, got {}",
            transform.transform_x(penalty_spot_x)
        );

        // Center circle edge: 9.15m from center (in CDF that's just 9.15)
        // Should stay at 9.15 (IFAB fixed)
        let center_circle_x = 9.15;
        assert!(
            (transform.transform_x(center_circle_x) - 9.15).abs() < 0.001,
            "Center circle edge: expected 9.15, got {}",
            transform.transform_x(center_circle_x)
        );

        // Goal post: 3.66m from center (half of 7.32m goal width)
        // Should stay at 3.66 (IFAB fixed)
        let goal_post_y = 3.66;
        assert!(
            (transform.transform_y(goal_post_y) - 3.66).abs() < 0.001,
            "Goal post: expected 3.66, got {}",
            transform.transform_y(goal_post_y)
        );
    }

    #[test]
    fn test_dimension_transform_roundtrip() {
        // Transform A -> B -> A should restore original
        let transform_ab = DimensionTransform::new(105.0, 68.0, 110.0, 70.0);
        let transform_ba = DimensionTransform::new(110.0, 70.0, 105.0, 68.0);

        let test_points = [
            (0.0, 0.0),
            (25.0, 15.0),
            (-40.0, -30.0),
            (52.5, 34.0),
            (-52.5, -34.0),
        ];

        for (orig_x, orig_y) in test_points {
            let (mid_x, mid_y, _) = transform_ab.transform(orig_x, orig_y, 0.0);
            let (back_x, back_y, _) = transform_ba.transform(mid_x, mid_y, 0.0);

            assert!(
                (back_x - orig_x).abs() < 0.01,
                "Roundtrip X: {} -> {} -> {}, expected {}",
                orig_x,
                mid_x,
                back_x,
                orig_x
            );
            assert!(
                (back_y - orig_y).abs() < 0.01,
                "Roundtrip Y: {} -> {} -> {}, expected {}",
                orig_y,
                mid_y,
                back_y,
                orig_y
            );
        }
    }

    #[test]
    fn test_dimension_transform_combined_with_coordinate() {
        // Transform 110x70 -> 105x68, then CDF -> Tracab
        let dim_transform = DimensionTransform::new(110.0, 70.0, 105.0, 68.0);

        // Start: corner at (-55, -35) on 110x70 pitch
        let x1 = -55.0;
        let y1 = -35.0;

        // After dimension transform: should be at corner of 105x68 pitch
        let x2 = dim_transform.transform_x(x1);
        let y2 = dim_transform.transform_y(y1);

        assert!(
            (x2 - (-52.5)).abs() < 0.001,
            "Dim transform X: expected -52.5, got {}",
            x2
        );
        assert!(
            (y2 - (-34.0)).abs() < 0.001,
            "Dim transform Y: expected -34, got {}",
            y2
        );

        // Then to Tracab (centimeters)
        let x3 = x2 * 100.0;
        let y3 = y2 * 100.0;

        assert!(
            (x3 - (-5250.0)).abs() < 0.1,
            "Tracab X: expected -5250, got {}",
            x3
        );
        assert!(
            (y3 - (-3400.0)).abs() < 0.1,
            "Tracab Y: expected -3400, got {}",
            y3
        );
    }

    // ========================================================================
    // Unit enum tests
    // ========================================================================

    #[test]
    fn test_unit_for_coordinate_system() {
        assert_eq!(Unit::for_coordinate_system(CoordinateSystem::Cdf), Unit::Meters);
        assert_eq!(Unit::for_coordinate_system(CoordinateSystem::Tracab), Unit::Centimeters);
        assert_eq!(Unit::for_coordinate_system(CoordinateSystem::Kloppy), Unit::Normed);
        assert_eq!(Unit::for_coordinate_system(CoordinateSystem::Opta), Unit::Normed);
        assert_eq!(Unit::for_coordinate_system(CoordinateSystem::SportVu), Unit::Meters);
        assert_eq!(Unit::for_coordinate_system(CoordinateSystem::SportecEvent), Unit::Meters);
    }

    #[test]
    fn test_unit_to_meters_factor() {
        assert!((Unit::Meters.to_meters_factor() - 1.0).abs() < 0.001);
        assert!((Unit::Centimeters.to_meters_factor() - 0.01).abs() < 0.001);
        assert!((Unit::Yards.to_meters_factor() - 0.9144).abs() < 0.001);
        assert!((Unit::Feet.to_meters_factor() - 0.3048).abs() < 0.001);
    }
}
