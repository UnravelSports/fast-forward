use crate::error::KloppyError;

/// Coordinate system for tracking data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateSystem {
    /// Common Data Format: origin at center
    /// X ∈ [-pitch_length/2, +pitch_length/2]
    /// Y ∈ [-pitch_width/2, +pitch_width/2]
    Cdf,
}

impl CoordinateSystem {
    pub fn from_str(s: &str) -> Result<Self, KloppyError> {
        match s.to_lowercase().as_str() {
            "cdf" => Ok(CoordinateSystem::Cdf),
            _ => Err(KloppyError::UnsupportedCoordinates(s.to_string())),
        }
    }
}

/// Trait for coordinate transformation
pub trait CoordinateTransform {
    /// Transform coordinates to the target coordinate system
    fn transform(&self, x: f32, y: f32, z: f32, target: CoordinateSystem) -> (f32, f32, f32);

    /// Returns the native coordinate system of this provider
    fn native_system(&self) -> CoordinateSystem;
}

/// SecondSpectrum coordinate transformer
/// SecondSpectrum already uses CDF (center origin), so no transformation needed
pub struct SecondSpectrumCoordinates;

impl CoordinateTransform for SecondSpectrumCoordinates {
    fn transform(&self, x: f32, y: f32, z: f32, target: CoordinateSystem) -> (f32, f32, f32) {
        match target {
            CoordinateSystem::Cdf => (x, y, z), // No-op, already in CDF
        }
    }

    fn native_system(&self) -> CoordinateSystem {
        CoordinateSystem::Cdf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinate_system_from_str_cdf() {
        let result = CoordinateSystem::from_str("cdf");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), CoordinateSystem::Cdf);
    }

    #[test]
    fn test_coordinate_system_from_str_uppercase() {
        let result = CoordinateSystem::from_str("CDF");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), CoordinateSystem::Cdf);
    }

    #[test]
    fn test_coordinate_system_from_str_mixed_case() {
        let result = CoordinateSystem::from_str("Cdf");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), CoordinateSystem::Cdf);
    }

    #[test]
    fn test_coordinate_system_from_str_invalid() {
        let result = CoordinateSystem::from_str("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_secondspectrum_transform_cdf() {
        let transformer = SecondSpectrumCoordinates;
        let (x, y, z) = transformer.transform(10.0, 20.0, 1.5, CoordinateSystem::Cdf);
        assert_eq!(x, 10.0);
        assert_eq!(y, 20.0);
        assert_eq!(z, 1.5);
    }

    #[test]
    fn test_secondspectrum_native_system() {
        let transformer = SecondSpectrumCoordinates;
        assert_eq!(transformer.native_system(), CoordinateSystem::Cdf);
    }
}
