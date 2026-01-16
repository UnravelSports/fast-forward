/// Standardized player positions across all providers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    // Goalkeeper
    GK,

    // Defenders
    LB,
    RB,
    LCB,
    CB,
    RCB,
    LWB,
    RWB,

    // Defensive Midfielders
    LDM,
    CDM,
    RDM,

    // Central Midfielders
    LCM,
    CM,
    RCM,

    // Attacking Midfielders
    LAM,
    CAM,
    RAM,

    // Wide Midfielders
    LW,
    RW,
    LM,
    RM,

    // Attackers
    LF,
    ST,
    RF,
    CF,

    // Referees
    REF,    // Main Referee
    AREF,   // Assistant Referee
    VAR,    // Video Assistant Referee
    AVAR,   // Assistant VAR
    FOURTH, // Fourth Official (outputs as "4TH")

    // Special
    SUB,
    Unknown,
}

impl Position {
    pub fn as_str(&self) -> &'static str {
        match self {
            // Goalkeeper
            Position::GK => "GK",

            // Defenders
            Position::LB => "LB",
            Position::RB => "RB",
            Position::LCB => "LCB",
            Position::CB => "CB",
            Position::RCB => "RCB",
            Position::LWB => "LWB",
            Position::RWB => "RWB",

            // Defensive Midfielders
            Position::LDM => "LDM",
            Position::CDM => "CDM",
            Position::RDM => "RDM",

            // Central Midfielders
            Position::LCM => "LCM",
            Position::CM => "CM",
            Position::RCM => "RCM",

            // Attacking Midfielders
            Position::LAM => "LAM",
            Position::CAM => "CAM",
            Position::RAM => "RAM",

            // Wide Midfielders
            Position::LW => "LW",
            Position::RW => "RW",
            Position::LM => "LM",
            Position::RM => "RM",

            // Attackers
            Position::LF => "LF",
            Position::ST => "ST",
            Position::RF => "RF",
            Position::CF => "CF",

            // Referees
            Position::REF => "REF",
            Position::AREF => "AREF",
            Position::VAR => "VAR",
            Position::AVAR => "AVAR",
            Position::FOURTH => "4TH",

            // Special
            Position::SUB => "SUB",
            Position::Unknown => "UNK",
        }
    }

    /// Parse referee role from Sportec format
    pub fn from_sportec_referee_role(role: &str) -> Self {
        match role.to_lowercase().as_str() {
            "referee" => Position::REF,
            "firstassistant" | "secondassistant" => Position::AREF,
            "fourthofficial" => Position::FOURTH,
            "videoreferee" => Position::VAR,
            "videorefereeassistant" => Position::AVAR,
            _ => Position::Unknown,
        }
    }

    /// Parse position from Sportec format
    pub fn from_sportec(raw: &str) -> Self {
        match raw.to_uppercase().as_str() {
            "GK" => Position::GK,
            "LB" => Position::LB,
            "RB" => Position::RB,
            "LCB" => Position::LCB,
            "CB" => Position::CB,
            "RCB" => Position::RCB,
            "LWB" => Position::LWB,
            "RWB" => Position::RWB,
            "LDM" => Position::LDM,
            "CDM" | "DM" => Position::CDM,
            "RDM" => Position::RDM,
            "LCM" | "HLM" => Position::LCM, // HLM = Half-left midfield
            "CM" => Position::CM,
            "RCM" | "HRM" => Position::RCM, // HRM = Half-right midfield
            "LAM" => Position::LAM,
            "CAM" | "AM" => Position::CAM,
            "RAM" => Position::RAM,
            "LW" => Position::LW,
            "RW" => Position::RW,
            "LM" => Position::LM,
            "RM" => Position::RM,
            "LF" => Position::LF,
            "ST" | "FW" => Position::ST,
            "RF" => Position::RF,
            "CF" => Position::CF,
            "SUB" => Position::SUB,
            _ => Position::Unknown,
        }
    }

    /// Parse position from SecondSpectrum format
    pub fn from_secondspectrum(raw: &str) -> Self {
        match raw.to_uppercase().as_str() {
            "GK" => Position::GK,
            "LB" => Position::LB,
            "RB" => Position::RB,
            "LCB" => Position::LCB,
            "CB" => Position::CB,
            "RCB" => Position::RCB,
            "LWB" => Position::LWB,
            "RWB" => Position::RWB,
            "LDM" => Position::LDM,
            "CDM" | "DM" => Position::CDM,
            "RDM" => Position::RDM,
            "LCM" => Position::LCM,
            "CM" => Position::CM,
            "RCM" => Position::RCM,
            "LAM" => Position::LAM,
            "CAM" | "AM" => Position::CAM,
            "RAM" => Position::RAM,
            "LW" => Position::LW,
            "RW" => Position::RW,
            "LM" => Position::LM,
            "RM" => Position::RM,
            "LF" => Position::LF,
            "ST" | "FW" => Position::ST,
            "RF" => Position::RF,
            "CF" => Position::CF,
            "SUB" => Position::SUB,
            _ => Position::Unknown,
        }
    }

    /// Parse position from SkillCorner format
    pub fn from_skillcorner(raw: &str) -> Self {
        match raw.to_uppercase().as_str() {
            "GK" => Position::GK,
            "LB" => Position::LB,
            "RB" => Position::RB,
            "LCB" => Position::LCB,
            "CB" => Position::CB,
            "RCB" => Position::RCB,
            "LWB" => Position::LWB,
            "RWB" => Position::RWB,
            "LDM" => Position::LDM,
            "CDM" | "DM" => Position::CDM,
            "RDM" => Position::RDM,
            "LCM" => Position::LCM,
            "CM" => Position::CM,
            "RCM" => Position::RCM,
            "LAM" => Position::LAM,
            "CAM" | "AM" => Position::CAM,
            "RAM" => Position::RAM,
            "LW" => Position::LW,
            "RW" => Position::RW,
            "LM" => Position::LM,
            "RM" => Position::RM,
            "LF" => Position::LF,
            "ST" | "CF" => Position::CF,
            "RF" => Position::RF,
            _ => Position::Unknown,
        }
    }

    /// Parse position from Tracab format
    /// Tracab uses single-letter codes: G (Goalkeeper), D (Defender), M (Midfielder), A (Attacker), S (Substitute)
    pub fn from_tracab(raw: &str) -> Self {
        match raw.to_uppercase().as_str() {
            "G" => Position::GK,
            "D" => Position::CB,  // Generic defender
            "M" => Position::CM,  // Generic midfielder
            "A" => Position::ST,  // Generic attacker/forward
            "S" => Position::SUB, // Substitute
            "O" => Position::SUB, // Off (subbed off)
            _ => Position::Unknown,
        }
    }

    /// Parse position from GradientSports (PFF) format
    /// PFF uses position group codes like: GK, RB, LB, RCB, LCB, CB, CDM, CM, CAM, AM, RW, LW, CF, ST
    pub fn from_gradientsports(raw: &str) -> Self {
        match raw.to_uppercase().as_str() {
            "GK" => Position::GK,
            "LB" => Position::LB,
            "RB" => Position::RB,
            "LCB" => Position::LCB,
            "CB" => Position::CB,
            "RCB" => Position::RCB,
            "LWB" => Position::LWB,
            "RWB" => Position::RWB,
            "LDM" => Position::LDM,
            "CDM" | "DM" => Position::CDM,
            "RDM" => Position::RDM,
            "LCM" => Position::LCM,
            "CM" => Position::CM,
            "RCM" => Position::RCM,
            "LAM" => Position::LAM,
            "CAM" | "AM" => Position::CAM,
            "RAM" => Position::RAM,
            "LW" => Position::LW,
            "RW" => Position::RW,
            "LM" => Position::LM,
            "RM" => Position::RM,
            "LF" => Position::LF,
            "ST" | "FW" => Position::ST,
            "RF" => Position::RF,
            "CF" => Position::CF,
            _ => Position::Unknown,
        }
    }

    /// Parse position from StatsPerform (Opta) format
    /// StatsPerform uses position names like "Goalkeeper", "Defender", "Midfielder", "Striker", "Substitute"
    /// with optional position_side values like "Left", "Right", "Centre", "Left/Centre", "Centre/Right"
    pub fn from_statsperform(position: &str, position_side: Option<&str>) -> Self {
        match position.to_lowercase().as_str() {
            "goalkeeper" => Position::GK,
            "defender" => Self::statsperform_defender_position(position_side),
            "midfielder" => Self::statsperform_midfielder_position(position_side),
            "striker" | "attacker" => Self::statsperform_attacker_position(position_side),
            "substitute" => Position::SUB,
            _ => Position::Unknown,
        }
    }

    /// Map StatsPerform defender position based on position_side
    fn statsperform_defender_position(side: Option<&str>) -> Self {
        match side.unwrap_or("").to_lowercase().as_str() {
            "left" => Position::LB,
            "right" => Position::RB,
            "centre" | "center" => Position::CB,
            "left/centre" | "left/center" => Position::LCB,
            "centre/right" | "center/right" => Position::RCB,
            _ => Position::CB,
        }
    }

    /// Map StatsPerform midfielder position based on position_side
    fn statsperform_midfielder_position(side: Option<&str>) -> Self {
        match side.unwrap_or("").to_lowercase().as_str() {
            "left" => Position::LM,
            "right" => Position::RM,
            "centre" | "center" => Position::CM,
            "left/centre" | "left/center" => Position::LCM,
            "centre/right" | "center/right" => Position::RCM,
            _ => Position::CM,
        }
    }

    /// Map StatsPerform attacker/striker position based on position_side
    fn statsperform_attacker_position(side: Option<&str>) -> Self {
        match side.unwrap_or("").to_lowercase().as_str() {
            "left" | "left/centre" | "left/center" => Position::LW,
            "right" | "centre/right" | "center/right" => Position::RW,
            "centre" | "center" => Position::ST,
            _ => Position::ST,
        }
    }

    /// Parse official/referee role from StatsPerform format
    /// StatsPerform uses types like "Main", "Assistant referee 1", "Fourth official", etc.
    pub fn from_statsperform_official(official_type: &str) -> Self {
        match official_type.to_lowercase().as_str() {
            "main" => Position::REF,
            "assistant referee 1" | "assistant referee 2" => Position::AREF,
            "fourth official" => Position::FOURTH,
            "video assistant referee" => Position::VAR,
            "assistant video assistant referee" => Position::AVAR,
            _ => Position::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_as_str() {
        assert_eq!(Position::GK.as_str(), "GK");
        assert_eq!(Position::LB.as_str(), "LB");
        assert_eq!(Position::ST.as_str(), "ST");
        assert_eq!(Position::SUB.as_str(), "SUB");
        assert_eq!(Position::Unknown.as_str(), "UNK");
    }

    #[test]
    fn test_from_secondspectrum() {
        assert_eq!(Position::from_secondspectrum("GK"), Position::GK);
        assert_eq!(Position::from_secondspectrum("gk"), Position::GK);
        assert_eq!(Position::from_secondspectrum("FW"), Position::ST);
        assert_eq!(Position::from_secondspectrum("SUB"), Position::SUB);
        assert_eq!(Position::from_secondspectrum("UNKNOWN"), Position::Unknown);
    }

    #[test]
    fn test_from_skillcorner() {
        assert_eq!(Position::from_skillcorner("GK"), Position::GK);
        assert_eq!(Position::from_skillcorner("CF"), Position::CF);
        assert_eq!(Position::from_skillcorner("DM"), Position::CDM);
    }

    #[test]
    fn test_referee_positions() {
        assert_eq!(Position::REF.as_str(), "REF");
        assert_eq!(Position::AREF.as_str(), "AREF");
        assert_eq!(Position::VAR.as_str(), "VAR");
        assert_eq!(Position::AVAR.as_str(), "AVAR");
        assert_eq!(Position::FOURTH.as_str(), "4TH");
    }

    #[test]
    fn test_from_sportec_referee_role() {
        assert_eq!(Position::from_sportec_referee_role("referee"), Position::REF);
        assert_eq!(Position::from_sportec_referee_role("firstAssistant"), Position::AREF);
        assert_eq!(Position::from_sportec_referee_role("secondAssistant"), Position::AREF);
        assert_eq!(Position::from_sportec_referee_role("fourthOfficial"), Position::FOURTH);
        assert_eq!(Position::from_sportec_referee_role("videoReferee"), Position::VAR);
        assert_eq!(Position::from_sportec_referee_role("videoRefereeAssistant"), Position::AVAR);
        assert_eq!(Position::from_sportec_referee_role("unknown"), Position::Unknown);
    }

    #[test]
    fn test_from_sportec() {
        assert_eq!(Position::from_sportec("GK"), Position::GK);
        assert_eq!(Position::from_sportec("RW"), Position::RW);
        assert_eq!(Position::from_sportec("HLM"), Position::LCM);
        assert_eq!(Position::from_sportec("HRM"), Position::RCM);
        assert_eq!(Position::from_sportec("CF"), Position::CF);
    }

    #[test]
    fn test_from_statsperform() {
        // Goalkeeper
        assert_eq!(Position::from_statsperform("Goalkeeper", Some("Centre")), Position::GK);
        assert_eq!(Position::from_statsperform("Goalkeeper", None), Position::GK);

        // Defenders
        assert_eq!(Position::from_statsperform("Defender", Some("Left")), Position::LB);
        assert_eq!(Position::from_statsperform("Defender", Some("Right")), Position::RB);
        assert_eq!(Position::from_statsperform("Defender", Some("Centre")), Position::CB);
        assert_eq!(Position::from_statsperform("Defender", Some("Left/Centre")), Position::LCB);
        assert_eq!(Position::from_statsperform("Defender", Some("Centre/Right")), Position::RCB);
        assert_eq!(Position::from_statsperform("Defender", None), Position::CB);

        // Midfielders
        assert_eq!(Position::from_statsperform("Midfielder", Some("Left")), Position::LM);
        assert_eq!(Position::from_statsperform("Midfielder", Some("Right")), Position::RM);
        assert_eq!(Position::from_statsperform("Midfielder", Some("Centre")), Position::CM);
        assert_eq!(Position::from_statsperform("Midfielder", Some("Left/Centre")), Position::LCM);
        assert_eq!(Position::from_statsperform("Midfielder", Some("Centre/Right")), Position::RCM);

        // Strikers
        assert_eq!(Position::from_statsperform("Striker", Some("Centre")), Position::ST);
        assert_eq!(Position::from_statsperform("Striker", Some("Left/Centre")), Position::LW);
        assert_eq!(Position::from_statsperform("Striker", Some("Centre/Right")), Position::RW);
        assert_eq!(Position::from_statsperform("Attacker", None), Position::ST);

        // Substitute
        assert_eq!(Position::from_statsperform("Substitute", None), Position::SUB);

        // Case insensitivity
        assert_eq!(Position::from_statsperform("goalkeeper", Some("centre")), Position::GK);
        assert_eq!(Position::from_statsperform("DEFENDER", Some("LEFT")), Position::LB);
    }

    #[test]
    fn test_from_statsperform_official() {
        assert_eq!(Position::from_statsperform_official("Main"), Position::REF);
        assert_eq!(Position::from_statsperform_official("Assistant referee 1"), Position::AREF);
        assert_eq!(Position::from_statsperform_official("Assistant referee 2"), Position::AREF);
        assert_eq!(Position::from_statsperform_official("Fourth official"), Position::FOURTH);
        assert_eq!(Position::from_statsperform_official("Video assistant referee"), Position::VAR);
        assert_eq!(Position::from_statsperform_official("Assistant video assistant referee"), Position::AVAR);
        assert_eq!(Position::from_statsperform_official("unknown"), Position::Unknown);
        // Case insensitivity
        assert_eq!(Position::from_statsperform_official("MAIN"), Position::REF);
        assert_eq!(Position::from_statsperform_official("fourth OFFICIAL"), Position::FOURTH);
    }
}
