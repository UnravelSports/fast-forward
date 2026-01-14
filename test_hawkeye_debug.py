"""Debug script to test HawkEye single file loading."""

from kloppy_light import hawkeye
import polars as pl
from pathlib import Path

# Test data paths
BALL_FILE = "tests/files/hawkeye_2_46.football.samples.ball"
PLAYER_FILE = "tests/files/hawkeye_2_46.football.samples.centroids"
META_JSON = "tests/files/hawkeye_meta.json"

print("=" * 80)
print("Test 1: Lazy loading with single file (the failing case)")
print("=" * 80)

tracking_lazy, metadata_df, team_df, player_df = hawkeye.load_tracking(
    ball_data=BALL_FILE,
    player_data=PLAYER_FILE,
    meta_data=META_JSON,
    layout="long",
    lazy=True
)

print(f"Lazy loader created: {tracking_lazy}")
print(f"Metadata shape: {metadata_df.shape}")
print(f"Team shape: {team_df.shape}")
print(f"Player shape: {player_df.shape}")

print("\nNow filtering by period_id == 2...")
result = tracking_lazy.filter(pl.col("period_id") == 2).collect()
print(f"Result shape after filter: {result.shape}")
print(f"Result preview:")
print(result.head(10))

print("\n" + "=" * 80)
print("Test 2: Collect without filter")
print("=" * 80)

tracking_lazy2, _, _, _ = hawkeye.load_tracking(
    ball_data=BALL_FILE,
    player_data=PLAYER_FILE,
    meta_data=META_JSON,
    layout="long",
    lazy=True
)

result2 = tracking_lazy2.collect()
print(f"Result shape: {result2.shape}")
print(f"Unique period_ids: {sorted(result2['period_id'].unique().to_list())}")
print(f"Result preview:")
print(result2.head(10))

print("\n" + "=" * 80)
print("Test 3: Eager loading (the working case)")
print("=" * 80)

tracking_df, metadata_df, team_df, player_df = hawkeye.load_tracking(
    ball_data=BALL_FILE,
    player_data=PLAYER_FILE,
    meta_data=META_JSON,
    layout="long",
    lazy=False
)

print(f"Tracking shape: {tracking_df.shape}")
print(f"Unique period_ids: {sorted(tracking_df['period_id'].unique().to_list())}")
print(f"Result preview:")
print(tracking_df.head(10))
print(f"\nFiltered by period_id == 2:")
filtered = tracking_df.filter(pl.col("period_id") == 2)
print(f"Filtered shape: {filtered.shape}")
print(filtered.head(10))
