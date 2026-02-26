"""Centralized test file path configuration.

This module provides all test data file paths in one place to avoid
duplication across test files.
"""

from pathlib import Path

# Base directory for all test data files
DATA_DIR = Path(__file__).parent / "files"

# =============================================================================
# SecondSpectrum
# =============================================================================
SS_RAW_ANON = str(DATA_DIR / "secondspectrum_tracking_anon.jsonl")
SS_META_ANON = str(DATA_DIR / "secondspectrum_meta_anon.json")
SS_RAW_NULL_BALL = str(DATA_DIR / "secondspectrum_tracking_null_ball.jsonl")
SS_RAW_FAKE = str(DATA_DIR / "second_spectrum_fake_data.jsonl")
SS_RAW_FAKE_UTF8SIG = str(DATA_DIR / "second_spectrum_fake_data_utf8sig.jsonl")
SS_META_FAKE = str(DATA_DIR / "second_spectrum_fake_metadata.json")
SS_META_FAKE_XML = str(DATA_DIR / "second_spectrum_fake_metadata.xml")
SS_META_FAKE_XML_BOM = str(DATA_DIR / "second_spectrum_fake_metadata_bom.xml")

# =============================================================================
# SkillCorner
# =============================================================================
SC_RAW = str(DATA_DIR / "skillcorner_tracking.jsonl")
SC_META = str(DATA_DIR / "skillcorner_meta.json")
SC_RAW_BOUNDARY = str(DATA_DIR / "skillcorner_boundary_tracking.jsonl")
SC_META_BOUNDARY = str(DATA_DIR / "skillcorner_boundary_meta.json")

# =============================================================================
# Sportec
# =============================================================================
SP_META = str(DATA_DIR / "sportec_meta.xml")
SP_META_BOM = str(DATA_DIR / "sportec_meta_bom.xml")
SP_RAW = str(DATA_DIR / "sportec_positional.xml")
SP_RAW_BOM = str(DATA_DIR / "sportec_positional_bom.xml")
SP_RAW_W_REF = str(DATA_DIR / "sportec_positional_w_referee.xml")

# =============================================================================
# HawkEye
# =============================================================================
HE_META_JSON = str(DATA_DIR / "hawkeye_meta.json")
HE_META_XML = str(DATA_DIR / "hawkeye_meta.xml")
HE_META_XML_BOM = str(DATA_DIR / "hawkeye_meta_bom.xml")
HE_BALL_1 = str(DATA_DIR / "hawkeye_1_1.football.samples.ball")
HE_BALL_2 = str(DATA_DIR / "hawkeye_2_46.football.samples.ball")
HE_PLAYER_1 = str(DATA_DIR / "hawkeye_1_1.football.samples.centroids")
HE_PLAYER_2 = str(DATA_DIR / "hawkeye_2_46.football.samples.centroids")
HE_BALL_FILES = [HE_BALL_1, HE_BALL_2]
HE_PLAYER_FILES = [HE_PLAYER_1, HE_PLAYER_2]

# =============================================================================
# Signality
# =============================================================================
SIG_META = str(DATA_DIR / "signality_meta_data.json")
SIG_VENUE = str(DATA_DIR / "signality_venue_information.json")
SIG_RAW_P1 = str(DATA_DIR / "signality_p1_raw_data_subset.json")
SIG_RAW_P2 = str(DATA_DIR / "signality_p2_raw_data_subset.json")
SIG_RAW_FILES = [SIG_RAW_P1, SIG_RAW_P2]

# =============================================================================
# Respovision
# =============================================================================
RV_RAW = str(DATA_DIR / "respovision_tracking.jsonl")

# =============================================================================
# Tracab
# =============================================================================
TR_META_XML = str(DATA_DIR / "tracab_meta.xml")
TR_META_XML_BOM = str(DATA_DIR / "tracab_meta_bom.xml")
TR_META_JSON = str(DATA_DIR / "tracab_meta.json")
TR_META_XML_2 = str(DATA_DIR / "tracab_meta_2.xml")
TR_META_XML_3 = str(DATA_DIR / "tracab_meta_3.xml")
TR_META_XML_4 = str(DATA_DIR / "tracab_meta_4.xml")
TR_RAW_DAT = str(DATA_DIR / "tracab_raw.dat")
TR_RAW_JSON = str(DATA_DIR / "tracab_raw.json")

# =============================================================================
# StatsPerform
# =============================================================================
STP_RAW_MA25 = str(DATA_DIR / "statsperform_tracking_ma25.txt")
STP_META_JSON = str(DATA_DIR / "statsperform_tracking_ma1.json")
STP_META_XML = str(DATA_DIR / "statsperform_tracking_ma1.xml")
STP_META_XML_BOM = str(DATA_DIR / "statsperform_tracking_ma1_bom.xml")

# =============================================================================
# CDF (Common Data Format)
# =============================================================================
CDF_RAW = str(DATA_DIR / "cdf_tracking.jsonl")
CDF_META = str(DATA_DIR / "cdf_metadata.json")

# =============================================================================
# GradientSports (PFF)
# =============================================================================
GS_RAW = str(DATA_DIR / "pff_10517.jsonl")
GS_META = str(DATA_DIR / "pff_metadata_10517.json")
GS_ROSTER = str(DATA_DIR / "pff_rosters_10517.json")
GS_RAW_2 = str(DATA_DIR / "pff_3812.jsonl")
GS_META_2 = str(DATA_DIR / "pff_metadata_3812.json")
GS_ROSTER_2 = str(DATA_DIR / "pff_rosters_3812.json")
