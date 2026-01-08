mod metadata;
mod player;
mod team;
mod tracking_long;
mod tracking_long_ball;
mod tracking_wide;

pub use metadata::build_metadata_df;
pub use player::build_player_df;
pub use team::build_team_df;

use crate::error::KloppyError;
use crate::models::StandardFrame;
use polars::prelude::*;

/// Layout options for tracking DataFrame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Ball as row with team_id="ball", player_id="ball"
    Long,
    /// Ball in separate columns, only player rows
    LongBall,
    /// One row per frame, player_id in column names
    Wide,
}

impl Layout {
    pub fn from_str(s: &str) -> Result<Self, KloppyError> {
        match s.to_lowercase().as_str() {
            "long" => Ok(Layout::Long),
            "long_ball" => Ok(Layout::LongBall),
            "wide" => Ok(Layout::Wide),
            _ => Err(KloppyError::UnsupportedLayout(s.to_string())),
        }
    }
}

/// Build tracking DataFrame based on layout
pub fn build_tracking_df(frames: &[StandardFrame], layout: Layout, game_id: Option<&str>) -> Result<DataFrame, KloppyError> {
    match layout {
        Layout::Long => tracking_long::build(frames, game_id),
        Layout::LongBall => tracking_long_ball::build(frames, game_id),
        Layout::Wide => tracking_wide::build(frames, game_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_from_str_long() {
        assert_eq!(Layout::from_str("long").unwrap(), Layout::Long);
    }

    #[test]
    fn test_layout_from_str_long_ball() {
        assert_eq!(Layout::from_str("long_ball").unwrap(), Layout::LongBall);
    }

    #[test]
    fn test_layout_from_str_wide() {
        assert_eq!(Layout::from_str("wide").unwrap(), Layout::Wide);
    }

    #[test]
    fn test_layout_from_str_case_insensitive() {
        assert_eq!(Layout::from_str("LONG").unwrap(), Layout::Long);
        assert_eq!(Layout::from_str("Long_Ball").unwrap(), Layout::LongBall);
        assert_eq!(Layout::from_str("WIDE").unwrap(), Layout::Wide);
    }

    #[test]
    fn test_layout_from_str_invalid() {
        assert!(Layout::from_str("invalid").is_err());
        assert!(Layout::from_str("").is_err());
    }
}
