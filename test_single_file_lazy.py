"""Test loading a single specific file with lazy=True."""

from fastforward import hawkeye
import polars as pl

# Single file paths (not directory)
BALL_FILE = "tests/files/hawkeye_2_46.football.samples.ball"
PLAYER_FILE = "tests/files/hawkeye_2_46.football.samples.centroids"
META_JSON = "tests/files/hawkeye_meta.json"

print("=" * 80)
print("Test: Single file with lazy=True (user's failing scenario)")
print("=" * 80)

tracking_lazy, metadata_df, team_df, player_df = hawkeye.load_tracking(
    ball_data=BALL_FILE,
    player_data=PLAYER_FILE,
    meta_data=META_JSON,
    layout="long",
    lazy=True
)

print(f"Lazy loader: {tracking_lazy}")

print("\nCollect all without filter:")
all_data = tracking_lazy.collect()
print(f"Shape: {all_data.shape}")
print(f"Unique periods: {sorted(all_data['period_id'].unique().to_list())}")

print("\n" + "=" * 80)
print("Now filter by period_id == 2:")
print("=" * 80)

# Fresh lazy loader
tracking_lazy2, _, _, _ = hawkeye.load_tracking(
    ball_data=BALL_FILE,
    player_data=PLAYER_FILE,
    meta_data=META_JSON,
    layout="long",
    lazy=True
)

filtered = tracking_lazy2.filter(pl.col("period_id") == 2).collect()
print(f"Filtered shape: {filtered.shape}")
if len(filtered) > 0:
    print(f"Periods: {sorted(filtered['period_id'].unique().to_list())}")
    print(f"✓ SUCCESS - Got {len(filtered)} rows")
else:
    print("✗ FAILED - Empty dataframe!")

print("\n" + "=" * 80)
print("Comparison: Same file with lazy=False")
print("=" * 80)

tracking_df, _, _, _ = hawkeye.load_tracking(
    ball_data=BALL_FILE,
    player_data=PLAYER_FILE,
    meta_data=META_JSON,
    layout="long",
    lazy=False
)

print(f"Eager shape: {tracking_df.shape}")
print(f"Unique periods: {sorted(tracking_df['period_id'].unique().to_list())}")

filtered_eager = tracking_df.filter(pl.col("period_id") == 2)
print(f"Eager filtered: {filtered_eager.shape}")
