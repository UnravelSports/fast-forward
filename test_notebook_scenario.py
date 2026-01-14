"""Test the exact scenario from the notebook."""

from kloppy_light import hawkeye
import polars as pl
from pathlib import Path

# Data paths - simulating notebook structure
# In the notebook, they have a directory with multiple files
# Let's test with the test files
BALL_DIR = "tests/files"
PLAYER_DIR = "tests/files"
META_JSON = "tests/files/hawkeye_meta.json"

print("Available test files:")
ball_files = list(Path(BALL_DIR).glob("*.ball"))
player_files = list(Path(PLAYER_DIR).glob("*.centroids"))
print(f"Ball files: {[f.name for f in ball_files]}")
print(f"Player files: {[f.name for f in player_files]}")

print("\n" + "=" * 80)
print("Test 1: Loading directory with lazy=True (user's scenario)")
print("=" * 80)

tracking_lazy, metadata_df, team_df, player_df = hawkeye.load_tracking(
    ball_data=BALL_DIR,
    player_data=PLAYER_DIR,
    meta_data=META_JSON,
    layout="long",
    lazy=True
)

print(f"Lazy loader created: {tracking_lazy}")
print(f"Metadata shape: {metadata_df.shape}")
print(f"Team shape: {team_df.shape}")
print(f"Player shape: {player_df.shape}")

print("\nTest 1a: Collect all data (no filter)")
all_data = tracking_lazy.collect()
print(f"All data shape: {all_data.shape}")
print(f"Unique periods in data: {sorted(all_data['period_id'].unique().to_list())}")

print("\nTest 1b: Filter by period_id == 2")
# Create new lazy loader for fresh filter
tracking_lazy2, _, _, _ = hawkeye.load_tracking(
    ball_data=BALL_DIR,
    player_data=PLAYER_DIR,
    meta_data=META_JSON,
    layout="long",
    lazy=True
)
result = tracking_lazy2.filter(pl.col("period_id") == 2).collect()
print(f"Filtered result shape: {result.shape}")
print(f"Periods in filtered result: {sorted(result['period_id'].unique().to_list()) if len(result) > 0 else 'EMPTY!'}")

print("\nTest 1c: Filter by period_id == 1")
tracking_lazy3, _, _, _ = hawkeye.load_tracking(
    ball_data=BALL_DIR,
    player_data=PLAYER_DIR,
    meta_data=META_JSON,
    layout="long",
    lazy=True
)
result_p1 = tracking_lazy3.filter(pl.col("period_id") == 1).collect()
print(f"Period 1 result shape: {result_p1.shape}")

print("\n" + "=" * 80)
print("Test 2: Loading directory with lazy=False (comparison)")
print("=" * 80)

tracking_df, _, _, _ = hawkeye.load_tracking(
    ball_data=BALL_DIR,
    player_data=PLAYER_DIR,
    meta_data=META_JSON,
    layout="long",
    lazy=False
)

print(f"Eager tracking shape: {tracking_df.shape}")
print(f"Unique periods: {sorted(tracking_df['period_id'].unique().to_list())}")

filtered_eager = tracking_df.filter(pl.col("period_id") == 2)
print(f"Eager filtered by period_id == 2: {filtered_eager.shape}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
if len(result) > 0:
    print("✓ Lazy loading with filter WORKS!")
else:
    print("✗ Lazy loading with filter returns EMPTY dataframe")

if len(all_data) > 0:
    print("✓ Lazy loading without filter WORKS!")
else:
    print("✗ Lazy loading without filter returns EMPTY dataframe")
