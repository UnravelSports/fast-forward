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
            "cdf" | "secondspectrum" | "skillcorner" | "pff" | "sportec:tracking" | "hawkeye" => {
                Ok(CoordinateSystem::Cdf)
            }
            // Distinct coordinate systems
            "kloppy" => Ok(CoordinateSystem::Kloppy),
            "tracab" => Ok(CoordinateSystem::Tracab),
            "sportvu" => Ok(CoordinateSystem::SportVu),
            "sportec:event" => Ok(CoordinateSystem::SportecEvent),
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
}
