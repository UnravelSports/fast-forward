#!/usr/bin/env python3
"""Test FileLike with actual data files."""

import sys
from pathlib import Path

def test_with_real_data():
    """Test FileLike integration with real data files."""
    print("Testing FileLike with actual data...")

    from kloppy_light import secondspectrum

    test_dir = Path("tests/files")
    raw_path = test_dir / "secondspectrum_tracking_anon.jsonl"
    meta_path = test_dir / "secondspectrum_meta_anon.json"

    if not raw_path.exists() or not meta_path.exists():
        print("✗ Test files not found")
        return False

    # Test 1: String paths (existing behavior)
    print("\n1. Testing with string paths...")
    try:
        result = secondspectrum.load_tracking(
            str(raw_path),
            str(meta_path),
            only_alive=True
        )
        tracking_df, metadata_df, team_df, player_df = result
        print(f"   ✓ Loaded {len(tracking_df)} tracking rows")
        print(f"   ✓ Loaded {len(team_df)} teams, {len(player_df)} players")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 2: Path objects
    print("\n2. Testing with Path objects...")
    try:
        result = secondspectrum.load_tracking(
            raw_path,
            meta_path,
            only_alive=True
        )
        tracking_df, metadata_df, team_df, player_df = result
        print(f"   ✓ Loaded {len(tracking_df)} tracking rows")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 3: Bytes objects
    print("\n3. Testing with bytes...")
    try:
        with open(raw_path, "rb") as f:
            raw_bytes = f.read()
        with open(meta_path, "rb") as f:
            meta_bytes = f.read()

        result = secondspectrum.load_tracking(
            raw_bytes,
            meta_bytes,
            only_alive=True
        )
        tracking_df, metadata_df, team_df, player_df = result
        print(f"   ✓ Loaded {len(tracking_df)} tracking rows")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: File handles
    print("\n4. Testing with file handles...")
    try:
        with open(raw_path, "rb") as raw_file, open(meta_path, "rb") as meta_file:
            result = secondspectrum.load_tracking(
                raw_file,
                meta_file,
                only_alive=True
            )
            tracking_df, metadata_df, team_df, player_df = result
            print(f"   ✓ Loaded {len(tracking_df)} tracking rows")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Lazy loading with FileLike
    print("\n5. Testing lazy loading with Path objects...")
    try:
        lazy_loader, metadata_df, team_df, player_df = secondspectrum.load_tracking(
            raw_path,
            meta_path,
            only_alive=True,
            lazy=True
        )
        print(f"   ✓ Created lazy loader")
        print(f"   ✓ Loaded {len(team_df)} teams, {len(player_df)} players")

        # Collect the data
        tracking_df = lazy_loader.collect()
        print(f"   ✓ Collected {len(tracking_df)} tracking rows from lazy loader")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Test with SkillCorner
    print("\n6. Testing SkillCorner with Path objects...")
    try:
        from kloppy_light import skillcorner

        sc_raw = test_dir / "skillcorner_tracking.jsonl"
        sc_meta = test_dir / "skillcorner_meta.json"

        if sc_raw.exists() and sc_meta.exists():
            result = skillcorner.load_tracking(
                sc_raw,
                sc_meta,
                only_alive=True
            )
            tracking_df, metadata_df, team_df, player_df = result
            print(f"   ✓ SkillCorner: Loaded {len(tracking_df)} tracking rows")
        else:
            print("   ⚠ SkillCorner test files not found")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 7: Test with Sportec
    print("\n7. Testing Sportec with bytes...")
    try:
        from kloppy_light import sportec

        sp_raw = test_dir / "sportec_positional.xml"
        sp_meta = test_dir / "sportec_meta.xml"

        if sp_raw.exists() and sp_meta.exists():
            with open(sp_raw, "rb") as f:
                raw_bytes = f.read()
            with open(sp_meta, "rb") as f:
                meta_bytes = f.read()

            result = sportec.load_tracking(
                raw_bytes,
                meta_bytes,
                only_alive=True
            )
            tracking_df, metadata_df, team_df, player_df = result
            print(f"   ✓ Sportec: Loaded {len(tracking_df)} tracking rows")
        else:
            print("   ⚠ Sportec test files not found")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("All data tests passed! ✓")
    print("="*60)
    print("\nVerified FileLike support for:")
    print("  ✓ String paths")
    print("  ✓ Path objects")
    print("  ✓ Bytes objects")
    print("  ✓ File handles")
    print("  ✓ Lazy loading")
    print("  ✓ All three providers (SecondSpectrum, SkillCorner, Sportec)")

    return True

if __name__ == "__main__":
    success = test_with_real_data()
    sys.exit(0 if success else 1)
