"""Cache utilities using kloppy's FileLike for read/write anywhere.

This module provides caching functionality that can write to local filesystem,
S3, GCS, or any other storage backend supported by kloppy's adapter pattern.
"""

import hashlib
import json
import os
import platform
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import polars as pl
from kloppy.io import open_as_file

# Schema version - increment when DataFrame schema changes
# Increment this when cache format changes (e.g., adding metadata storage)
CACHE_SCHEMA_VERSION = "2"

# Environment variable name for cache directory
CACHE_DIR_ENV_VAR = "KLOPPY_LIGHT_CACHE_DIR"

# Global cache directory setting (None = use default)
_global_cache_dir: Optional[str] = None


# Type alias for cache result with metadata
CacheResult = Tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]


def set_cache_dir(path: Union[str, Path, None]) -> None:
    """Set the global cache directory.

    This affects all subsequent cache operations. The setting persists
    for the lifetime of the Python process.

    Args:
        path: Cache directory path or URI (e.g., "/path/to/cache" or "s3://bucket/cache").
              Pass None to reset to default behavior.

    Example:
        >>> import fastforward
        >>> fastforward.set_cache_dir("/my/cache/dir")
        >>> # All subsequent cache operations use this directory
        >>> fastforward.set_cache_dir(None)  # Reset to default
    """
    global _global_cache_dir
    _global_cache_dir = str(path) if path is not None else None


def get_cache_dir() -> Path:
    """Get the current cache directory.

    Returns the cache directory in order of precedence:
    1. Global setting via set_cache_dir()
    2. KLOPPY_LIGHT_CACHE_DIR environment variable
    3. Platform-specific default directory

    Returns:
        Path to the cache directory

    Example:
        >>> import fastforward
        >>> fastforward.get_cache_dir()
        PosixPath('/Users/user/Library/Caches/fast-forward')
    """
    if _global_cache_dir is not None:
        return Path(_global_cache_dir)

    if cache_dir := os.environ.get(CACHE_DIR_ENV_VAR):
        return Path(cache_dir)

    return get_default_cache_dir()


def get_default_cache_dir() -> Path:
    """Get default platform-specific cache directory.

    Returns platform-specific cache directory:
    - Linux: ~/.cache/fast-forward/
    - macOS: ~/Library/Caches/fast-forward/
    - Windows: %LOCALAPPDATA%/fast-forward/cache/

    Note: Use get_cache_dir() to get the effective cache directory,
    which respects set_cache_dir() and KLOPPY_LIGHT_CACHE_DIR env var.
    """
    system = platform.system()
    if system == "Windows":
        base = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
        return base / "fast-forward" / "cache"
    elif system == "Darwin":
        return Path.home() / "Library" / "Caches" / "fast-forward"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(xdg_cache) / "fast-forward"


def compute_cache_key(
    raw_data: bytes,
    meta_data: bytes,
    config_str: str,
) -> str:
    """Generate SHA256 cache key from content + config.

    The cache key is based on:
    - SHA256 hash of raw tracking data content
    - SHA256 hash of metadata content
    - Configuration string (includes all params that affect output)
    - Schema version (to invalidate cache on schema changes)

    Returns first 16 characters of the hex digest.

    Note: For local files, prefer compute_cache_key_fast() which uses
    path + mtime + size instead of reading file contents.

    Args:
        raw_data: Raw tracking data bytes
        meta_data: Metadata bytes
        config_str: Configuration string (e.g., "long|cdf|static_home_away|True|True|include_empty_frames=False")
    """
    hasher = hashlib.sha256()
    hasher.update(raw_data)
    hasher.update(meta_data)
    hasher.update(f"{config_str}|{CACHE_SCHEMA_VERSION}".encode())
    return hasher.hexdigest()[:16]


def compute_cache_key_fast(
    raw_path: str,
    meta_path: str,
    config_str: str,
) -> str:
    """Generate cache key from path + mtime + size (no file reads).

    This is much faster than compute_cache_key() for local files because it
    only performs stat() calls instead of reading entire file contents.

    The cache key is based on:
    - Absolute path of raw data file
    - mtime (nanoseconds) and size of raw data file
    - Absolute path of metadata file
    - mtime (nanoseconds) and size of metadata file
    - Configuration string (includes all params that affect output)
    - Schema version (to invalidate cache on schema changes)

    Returns first 16 characters of the hex digest.

    Note: Only works for local files. For remote files (S3, GCS, HTTP),
    use compute_cache_key() with file contents.

    Args:
        raw_path: Path to raw tracking data file
        meta_path: Path to metadata file
        config_str: Configuration string (e.g., "long|cdf|static_home_away|True|True|include_empty_frames=False")
    """
    raw_p = Path(raw_path)
    meta_p = Path(meta_path)

    raw_stat = raw_p.stat()
    meta_stat = meta_p.stat()

    hasher = hashlib.sha256()
    hasher.update(str(raw_p.resolve()).encode())
    hasher.update(f"{raw_stat.st_mtime_ns}|{raw_stat.st_size}".encode())
    hasher.update(str(meta_p.resolve()).encode())
    hasher.update(f"{meta_stat.st_mtime_ns}|{meta_stat.st_size}".encode())
    hasher.update(f"{config_str}|{CACHE_SCHEMA_VERSION}".encode())
    return hasher.hexdigest()[:16]


def compute_cache_key_fast_multi(
    file_paths: List[str],
    meta_path: str,
    config_str: str,
) -> str:
    """Generate cache key from multiple file paths + mtime + size (no file reads).

    Used for providers like HawkEye that have multiple input files.

    Args:
        file_paths: List of file paths to include in the hash
        meta_path: Path to metadata file
        config_str: Configuration string to include (layout, coords, etc.)

    Returns:
        First 16 characters of the hex digest.
    """
    hasher = hashlib.sha256()

    # Add all file paths with their mtime and size
    for path_str in sorted(file_paths):  # Sort for deterministic order
        p = Path(path_str)
        stat = p.stat()
        hasher.update(str(p.resolve()).encode())
        hasher.update(f"{stat.st_mtime_ns}|{stat.st_size}".encode())

    # Add metadata file
    meta_p = Path(meta_path)
    meta_stat = meta_p.stat()
    hasher.update(str(meta_p.resolve()).encode())
    hasher.update(f"{meta_stat.st_mtime_ns}|{meta_stat.st_size}".encode())

    # Add config
    hasher.update(f"{config_str}|{CACHE_SCHEMA_VERSION}".encode())

    return hasher.hexdigest()[:16]


def get_cache_path(
    cache_key: str, provider: str, cache_dir: Optional[str] = None
) -> str:
    """Get cache file path/URI.

    Args:
        cache_key: The computed cache key
        provider: Provider name (e.g., "secondspectrum", "sportec")
        cache_dir: Optional custom cache directory (local path or S3/GCS URI).
                   If None, uses get_cache_dir() (respects set_cache_dir and env var).

    Returns:
        Full path/URI to the cache file
    """
    if cache_dir:
        # User-provided path (could be local or S3/GCS URI)
        base = cache_dir.rstrip("/")
        return f"{base}/{provider}/{cache_key}.parquet"
    else:
        # Use global cache directory (respects set_cache_dir and env var)
        local_dir = get_cache_dir() / provider
        local_dir.mkdir(parents=True, exist_ok=True)
        return str(local_dir / f"{cache_key}.parquet")


def cache_exists(cache_path: str) -> bool:
    """Check if cache file exists (works with local and remote).

    Args:
        cache_path: Path or URI to check

    Returns:
        True if cache file exists and is readable
    """
    # For local files, use Path.exists() for efficiency
    if not cache_path.startswith(("s3://", "gs://", "http://", "https://")):
        return Path(cache_path).exists()

    # For remote, try to open the file
    try:
        with open_as_file(cache_path, mode="rb") as f:
            return f is not None
    except Exception:
        return False


def _get_meta_path(cache_path: str) -> str:
    """Get the metadata sidecar file path for a cache file."""
    if cache_path.endswith(".parquet"):
        return cache_path[:-8] + ".meta.json"
    return cache_path + ".meta.json"


def write_cache(
    tracking_df: pl.DataFrame,
    cache_path: str,
    metadata_df: Optional[pl.DataFrame] = None,
    teams_df: Optional[pl.DataFrame] = None,
    players_df: Optional[pl.DataFrame] = None,
    periods_df: Optional[pl.DataFrame] = None,
) -> None:
    """Write DataFrame to cache location with optional metadata.

    Uses kloppy's write adapter to support local filesystem, S3, GCS, etc.
    Metadata DataFrames are stored in a sidecar JSON file.

    Args:
        tracking_df: Tracking DataFrame to cache
        cache_path: Destination path or URI
        metadata_df: Optional metadata DataFrame to store
        teams_df: Optional teams DataFrame to store
        players_df: Optional players DataFrame to store
        periods_df: Optional periods DataFrame to store
    """
    # For local files, ensure parent directory exists
    if not cache_path.startswith(("s3://", "gs://", "http://", "https://")):
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    # Write tracking DataFrame to Parquet
    buffer = BytesIO()
    tracking_df.write_parquet(buffer, compression="zstd")
    buffer.seek(0)

    # Write to destination using kloppy's adapter
    with open_as_file(cache_path, mode="wb") as f:
        f.write(buffer.read())

    # Write metadata to sidecar JSON file if provided
    if metadata_df is not None:
        meta_data = {
            "metadata": json.loads(metadata_df.write_json()),
            "teams": json.loads(teams_df.write_json()) if teams_df is not None else None,
            "players": json.loads(players_df.write_json()) if players_df is not None else None,
            "periods": json.loads(periods_df.write_json()) if periods_df is not None else None,
        }
        meta_path = _get_meta_path(cache_path)
        meta_json = json.dumps(meta_data)

        with open_as_file(meta_path, mode="wb") as f:
            f.write(meta_json.encode())


def read_cache(cache_path: str) -> Union[pl.LazyFrame, CacheResult]:
    """Read cache as LazyFrame with optional metadata.

    For local files, uses scan_parquet() for true lazy loading with
    predicate and projection pushdown.

    For remote files, downloads the entire file first (no lazy loading).

    Args:
        cache_path: Path or URI to read from

    Returns:
        If metadata is present: tuple of (LazyFrame, metadata_df, teams_df, players_df, periods_df)
        If no metadata: just the LazyFrame (for backwards compatibility with old cache files)
    """
    meta_path = _get_meta_path(cache_path)

    # For local files, use scan_parquet for true lazy loading
    if not cache_path.startswith(("s3://", "gs://", "http://", "https://")):
        lazy_frame = pl.scan_parquet(cache_path)

        # Check for sidecar metadata file
        if Path(meta_path).exists():
            with open(meta_path, "r") as f:
                meta_data = json.load(f)

            metadata_df = pl.read_json(BytesIO(json.dumps(meta_data["metadata"]).encode()))
            teams_df = pl.read_json(BytesIO(json.dumps(meta_data["teams"]).encode())) if meta_data["teams"] else pl.DataFrame()
            players_df = pl.read_json(BytesIO(json.dumps(meta_data["players"]).encode())) if meta_data["players"] else pl.DataFrame()
            periods_df = pl.read_json(BytesIO(json.dumps(meta_data["periods"]).encode())) if meta_data["periods"] else pl.DataFrame()

            return lazy_frame, metadata_df, teams_df, players_df, periods_df
        else:
            # No metadata file - return just LazyFrame
            return lazy_frame

    # For remote, we need to read into memory first
    with open_as_file(cache_path, mode="rb") as f:
        buffer = BytesIO(f.read())
    lazy_frame = pl.read_parquet(buffer).lazy()

    # Check for remote metadata file
    try:
        with open_as_file(meta_path, mode="rb") as f:
            meta_json = f.read()
        meta_data = json.loads(meta_json.decode())

        metadata_df = pl.read_json(BytesIO(json.dumps(meta_data["metadata"]).encode()))
        teams_df = pl.read_json(BytesIO(json.dumps(meta_data["teams"]).encode())) if meta_data["teams"] else pl.DataFrame()
        players_df = pl.read_json(BytesIO(json.dumps(meta_data["players"]).encode())) if meta_data["players"] else pl.DataFrame()
        periods_df = pl.read_json(BytesIO(json.dumps(meta_data["periods"]).encode())) if meta_data["periods"] else pl.DataFrame()

        return lazy_frame, metadata_df, teams_df, players_df, periods_df
    except Exception:
        # No metadata file - return just LazyFrame
        return lazy_frame


def clear_cache(provider: Optional[str] = None) -> int:
    """Clear local cache files.

    Args:
        provider: Optional provider name to clear only that provider's cache.
                 If None, clears all providers.

    Returns:
        Number of parquet files deleted (sidecar .meta.json files are also deleted)
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    count = 0
    if provider:
        provider_dir = cache_dir / provider
        if provider_dir.exists():
            for f in provider_dir.glob("*.parquet"):
                f.unlink()
                count += 1
                # Also delete sidecar metadata file
                meta_file = Path(_get_meta_path(str(f)))
                if meta_file.exists():
                    meta_file.unlink()
    else:
        for provider_dir in cache_dir.iterdir():
            if provider_dir.is_dir():
                for f in provider_dir.glob("*.parquet"):
                    f.unlink()
                    count += 1
                    # Also delete sidecar metadata file
                    meta_file = Path(_get_meta_path(str(f)))
                    if meta_file.exists():
                        meta_file.unlink()
    return count


def get_cache_size(provider: Optional[str] = None) -> int:
    """Get total local cache size in bytes.

    Args:
        provider: Optional provider name to get only that provider's cache size.
                 If None, gets total size across all providers.

    Returns:
        Total cache size in bytes
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    total = 0
    if provider:
        provider_dir = cache_dir / provider
        if provider_dir.exists():
            for f in provider_dir.glob("*.parquet"):
                total += f.stat().st_size
    else:
        for provider_dir in cache_dir.iterdir():
            if provider_dir.is_dir():
                for f in provider_dir.glob("*.parquet"):
                    total += f.stat().st_size
    return total
