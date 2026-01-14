"""Diagnose Tracab cache player_df issue."""
import json
import logging
import warnings
from pathlib import Path

# Enable logging to see cache hit messages
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from kloppy_light import tracab
from kloppy_light._cache import clear_cache, get_default_cache_dir

# Test with XML metadata that has NO player info
META_FILE = "tests/files/tracab_meta_2.xml"
RAW_FILE = "tests/files/tracab_raw.dat"

# Step 1: Clear tracab cache
print("=== Step 1: Clear tracab cache ===")
cleared = clear_cache("tracab")
print(f"Cleared {cleared} cache files")

# Step 2: First load
print(f"\n=== Step 2: First load (cache miss) with {META_FILE} ===")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dataset1 = tracab.load_tracking(
        RAW_FILE,
        META_FILE,
        lazy=True,
        cache=True,
    )
    if w:
        print(f"Warnings: {[str(x.message) for x in w]}")
print(f"players before collect: {len(dataset1.players)} rows")
print(dataset1.players)

# Step 3: Collect to trigger cache write
print("\n=== Step 3: Collect (triggers cache write) ===")
df = dataset1.tracking.collect()
print(f"Tracking rows: {len(df)}")

# Step 4: Inspect cache files
print("\n=== Step 4: Inspect cache files ===")
cache_dir = get_default_cache_dir() / "tracab"
print(f"Cache dir: {cache_dir}")
for f in sorted(cache_dir.glob("*")):
    print(f"  {f.name} ({f.stat().st_size} bytes)")
    if f.suffix == ".json":
        with open(f) as fh:
            meta = json.load(fh)
            players = meta.get('players')
            print(f"    players in cache: {len(players) if players else 0} rows")
            if players:
                print(f"    first player: {players[0]}")

# Step 5: Second load (should be cache hit)
print("\n=== Step 5: Second load (cache hit) ===")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dataset2 = tracab.load_tracking(
        RAW_FILE,
        META_FILE,
        lazy=True,
        cache=True,
    )
    if w:
        print(f"Warnings: {[str(x.message) for x in w]}")
print(f"players after cache hit: {len(dataset2.players)} rows")
print(dataset2.players)
