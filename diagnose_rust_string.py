"""Diagnose string filter extraction at Rust level.

This script tests whether string predicates are being extracted correctly
by checking what pushdown filters Rust detects.
"""

import polars as pl

# Import the sportec module to test with
from kloppy_light._kloppy_light import sportec

# Define test data paths
ST_META = "/Users/jbekkers/PycharmProjects/kloppy-light/data/sportec/MLS-MAT-0005WA_match_info.xml"
ST_TRACKING = "/Users/jbekkers/PycharmProjects/kloppy-light/data/sportec/MLS-MAT-0005WA_tracking.xml"

def test_filter_extraction():
    """Test whether Rust correctly extracts string filters."""

    # Test predicates
    predicates = [
        ("period_id == 1", pl.col("period_id") == 1),
        ("frame_id <= 50000", pl.col("frame_id") <= 50000),
        ("team_id == 'MLS-CLU-000065'", pl.col("team_id") == "MLS-CLU-000065"),
        ("player_id == 'MLS-OBJ-0007HS'", pl.col("player_id") == "MLS-OBJ-0007HS"),
    ]

    # First load without predicate to get baseline
    print("Loading baseline (no predicate)...")
    df_base, _, _, _, _ = sportec.load_tracking(
        open(ST_TRACKING, "rb").read(),
        open(ST_META, "rb").read(),
        layout="long",
        coordinates="cdf",
        orientation="static_home_away",
        only_alive=False,
        include_game_id=False,
        include_referees=False,
    )
    print(f"Baseline rows: {df_base.height}")

    # Test each predicate
    for name, pred in predicates:
        print(f"\nTesting: {name}")
        print(f"  Expression: {pred}")

        try:
            df, _, _, _, _ = sportec.load_tracking(
                open(ST_TRACKING, "rb").read(),
                open(ST_META, "rb").read(),
                layout="long",
                coordinates="cdf",
                orientation="static_home_away",
                only_alive=False,
                include_game_id=False,
                include_referees=False,
                predicate=pred,
            )
            print(f"  Result rows: {df.height}")

            # Check if filter was applied (rows < baseline)
            if df.height < df_base.height:
                print(f"  ✓ Filter applied in Rust ({df_base.height - df.height} rows filtered)")
            else:
                print(f"  ✗ Filter NOT applied in Rust (same row count)")

        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_filter_extraction()
