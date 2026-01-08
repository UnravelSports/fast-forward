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
}
