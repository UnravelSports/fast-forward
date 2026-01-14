#!/usr/bin/env python3
"""Script to help update test files from tuple unpacking to TrackingDataset API."""

import re
import sys
from pathlib import Path


def update_test_file(filepath: str) -> None:
    """Update a test file to use TrackingDataset API."""

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Pattern 1: Simple tuple unpacking in test methods
    # tracking_df, metadata_df, team_df, player_df = provider.load_tracking(...)
    # -> dataset = provider.load_tracking(..., lazy=False); access via dataset.tracking, etc

    patterns = [
        # Pattern 1a: All 4 components unpacked
        (
            r'(\w+_df),\s*(\w+_df),\s*(\w+_df),\s*(\w+_df)\s*=\s*(\w+)\.load_tracking\(([^)]+)\)',
            lambda m: f'dataset = {m.group(5)}.load_tracking({add_lazy_false(m.group(6))})'
        ),
        # Pattern 1b: Partial unpacking with underscores
        (
            r'([_\w]+),\s*([_\w]+),\s*([_\w]+),\s*([_\w]+)\s*=\s*(\w+)\.load_tracking\(([^)]+)\)',
            lambda m: handle_partial_unpack(m)
        ),
    ]

    # Apply patterns
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")
        return True
    else:
        print(f"No changes needed for {filepath}")
        return False


def add_lazy_false(args: str) -> str:
    """Add lazy=False to arguments if not present."""
    args = args.strip()
    if 'lazy=' not in args:
        # Remove trailing comma/whitespace
        args = args.rstrip().rstrip(',')
        args = args + ', lazy=False'
    return args


def handle_partial_unpack(match) -> str:
    """Handle partial tuple unpacking with underscores."""
    var1, var2, var3, var4, provider, args = match.groups()

    # Check if using underscores (like _, metadata_df, _, _)
    vars_list = [var1, var2, var3, var4]
    used_vars = [v for v in vars_list if v != '_']

    if len(used_vars) == 1:
        # Only one variable used - need to determine which one
        var_name = used_vars[0]
        if 'metadata' in var_name:
            return f'dataset = {provider}.load_tracking({add_lazy_false(args)})\n{var_name} = dataset.metadata'
        elif 'team' in var_name:
            return f'dataset = {provider}.load_tracking({add_lazy_false(args)})\n{var_name} = dataset.teams'
        elif 'player' in var_name:
            return f'dataset = {provider}.load_tracking({add_lazy_false(args)})\n{var_name} = dataset.players'
        elif 'tracking' in var_name:
            return f'dataset = {provider}.load_tracking({add_lazy_false(args)})\n{var_name} = dataset.tracking'

    # Default: full dataset
    return f'dataset = {provider}.load_tracking({add_lazy_false(args)})'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        update_test_file(filepath)
    else:
        print("Usage: python update_tests_script.py <test_file.py>")
        sys.exit(1)
