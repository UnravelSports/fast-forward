"""Diagnostic script to help debug HawkEye lazy loading issues.

This script helps identify why lazy loading might appear to return empty dataframes.
"""

from fastforward import hawkeye
import polars as pl
from pathlib import Path
import sys

def test_single_file():
    """Test loading a single file."""
    print("=" * 80)
    print("TEST 1: Single File Loading")
    print("=" * 80)

    ball_file = "tests/files/hawkeye_2_46.football.samples.ball"
    player_file = "tests/files/hawkeye_2_46.football.samples.centroids"
    meta_file = "tests/files/hawkeye_meta.json"

    print(f"Ball file: {ball_file}")
    print(f"Player file: {player_file}")
    print(f"Meta file: {meta_file}")

    # Check files exist
    if not Path(ball_file).exists():
        print(f"✗ Ball file does not exist!")
        return False
    if not Path(player_file).exists():
        print(f"✗ Player file does not exist!")
        return False

    print("✓ Files exist")

    # Test with lazy=True
    print("\nLoading with lazy=True...")
    tracking_lazy, metadata_df, team_df, player_df = hawkeye.load_tracking(
        ball_data=ball_file,
        player_data=player_file,
        meta_data=meta_file,
        layout="long",
        lazy=True
    )

    print(f"Lazy loader type: {type(tracking_lazy).__name__}")
    print(f"Metadata shape: {metadata_df.shape}")

    # Collect without filter
    print("\n1. Collecting without filter:")
    df_all = tracking_lazy.collect()
    print(f"   Result shape: {df_all.shape}")
    if len(df_all) > 0:
        print(f"   Unique periods: {sorted(df_all['period_id'].unique().to_list())}")
        print(f"   ✓ Got data!")
    else:
        print(f"   ✗ Empty dataframe!")
        return False

    # Create new lazy loader for filter test
    print("\n2. Creating new lazy loader and filtering by period_id == 2:")
    tracking_lazy2, _, _, _ = hawkeye.load_tracking(
        ball_data=ball_file,
        player_data=player_file,
        meta_data=meta_file,
        layout="long",
        lazy=True
    )
    df_filtered = tracking_lazy2.filter(pl.col("period_id") == 2).collect()
    print(f"   Result shape: {df_filtered.shape}")
    if len(df_filtered) > 0:
        print(f"   Unique periods: {sorted(df_filtered['period_id'].unique().to_list())}")
        print(f"   ✓ Filter works!")
        return True
    else:
        print(f"   ✗ Filter returned empty!")
        return False

def test_directory():
    """Test loading from directory."""
    print("\n" + "=" * 80)
    print("TEST 2: Directory Loading")
    print("=" * 80)

    ball_dir = "tests/files"
    player_dir = "tests/files"
    meta_file = "tests/files/hawkeye_meta.json"

    # Check what files are in directory
    ball_files = list(Path(ball_dir).glob("*.ball"))
    player_files = list(Path(player_dir).glob("*.centroids"))

    print(f"Ball directory: {ball_dir}")
    print(f"Found {len(ball_files)} ball files:")
    for f in sorted(ball_files):
        print(f"  - {f.name}")

    print(f"\nPlayer directory: {player_dir}")
    print(f"Found {len(player_files)} player files:")
    for f in sorted(player_files):
        print(f"  - {f.name}")

    if len(ball_files) == 0 or len(player_files) == 0:
        print("✗ No files found!")
        return False

    # Test with lazy=True
    print("\nLoading with lazy=True...")
    tracking_lazy, metadata_df, team_df, player_df = hawkeye.load_tracking(
        ball_data=ball_dir,
        player_data=player_dir,
        meta_data=meta_file,
        layout="long",
        lazy=True
    )

    print(f"Lazy loader: {tracking_lazy}")

    # Collect without filter
    print("\n1. Collecting without filter:")
    df_all = tracking_lazy.collect()
    print(f"   Result shape: {df_all.shape}")
    if len(df_all) > 0:
        print(f"   Unique periods: {sorted(df_all['period_id'].unique().to_list())}")
        print(f"   ✓ Got data!")
    else:
        print(f"   ✗ Empty dataframe!")
        return False

    # Filter by period
    print("\n2. Creating new lazy loader and filtering by period_id == 2:")
    tracking_lazy2, _, _, _ = hawkeye.load_tracking(
        ball_data=ball_dir,
        player_data=player_dir,
        meta_data=meta_file,
        layout="long",
        lazy=True
    )
    df_filtered = tracking_lazy2.filter(pl.col("period_id") == 2).collect()
    print(f"   Result shape: {df_filtered.shape}")
    if len(df_filtered) > 0:
        print(f"   Unique periods: {sorted(df_filtered['period_id'].unique().to_list())}")
        print(f"   ✓ Filter works!")
        return True
    else:
        print(f"   ✗ Filter returned empty!")
        return False

def test_common_mistakes():
    """Test common mistakes that might cause issues."""
    print("\n" + "=" * 80)
    print("TEST 3: Common Mistakes")
    print("=" * 80)

    ball_file = "tests/files/hawkeye_2_46.football.samples.ball"
    player_file = "tests/files/hawkeye_2_46.football.samples.centroids"
    meta_file = "tests/files/hawkeye_meta.json"

    print("Mistake 1: Not calling .collect()")
    print("-" * 40)
    tracking_lazy, _, _, _ = hawkeye.load_tracking(
        ball_data=ball_file,
        player_data=player_file,
        meta_data=meta_file,
        layout="long",
        lazy=True
    )

    # This returns a LazyTrackingLoader, not a DataFrame
    result_lazy = tracking_lazy.filter(pl.col("period_id") == 2)
    print(f"Without .collect(): type = {type(result_lazy).__name__}")
    print(f"  This is still a lazy loader, not data!")

    # This actually loads the data
    result_df = tracking_lazy.filter(pl.col("period_id") == 2).collect()
    print(f"With .collect(): type = {type(result_df).__name__}, shape = {result_df.shape}")
    print(f"  ✓ This is actual data!")

    print("\nMistake 2: Reusing the same lazy loader")
    print("-" * 40)
    tracking_lazy, _, _, _ = hawkeye.load_tracking(
        ball_data=ball_file,
        player_data=player_file,
        meta_data=meta_file,
        layout="long",
        lazy=True
    )

    # First collect
    df1 = tracking_lazy.filter(pl.col("period_id") == 2).collect()
    print(f"First collect: shape = {df1.shape}")

    # Reusing the same loader - this is fine!
    df2 = tracking_lazy.filter(pl.col("period_id") == 1).collect()
    print(f"Second collect (different filter): shape = {df2.shape}")
    print(f"  ✓ Lazy loaders can be reused!")

if __name__ == "__main__":
    success = True

    success &= test_single_file()
    success &= test_directory()
    test_common_mistakes()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if success:
        print("✓ All tests passed!")
        print("\nIf you're seeing empty dataframes in your code:")
        print("1. Make sure you're calling .collect() on the lazy loader")
        print("2. Check that your file paths are correct")
        print("3. Verify the files contain data for the period you're filtering")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        print("There may be an issue with the lazy loading implementation.")
        sys.exit(1)
