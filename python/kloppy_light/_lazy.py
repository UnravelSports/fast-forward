"""Lazy loading via Polars register_io_source.

This module provides functions to create real pl.LazyFrame objects for tracking data
using Polars' IO plugin system. The actual data parsing is deferred until .collect().
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union, TYPE_CHECKING

import polars as pl
from kloppy.io import FileLike, open_as_file

logger = logging.getLogger(__name__)


def _is_local_file(file_like: FileLike) -> bool:
    """Check if a FileLike is a local file (not a remote URL)."""
    if isinstance(file_like, (str, Path)):
        path_str = str(file_like)
        return not path_str.startswith(("s3://", "gs://", "http://", "https://"))
    # For file objects, assume local
    return hasattr(file_like, "name") and not str(getattr(file_like, "name", "")).startswith(
        ("s3://", "gs://", "http://", "https://")
    )


# Type alias for cache result with metadata
CacheLazyResult = Tuple[
    pl.LazyFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame
]


def create_lazy_tracking(
    provider: str,
    raw_data: FileLike,
    meta_data: FileLike,
    schema: Dict[str, pl.DataType],
    layout: str,
    coordinates: str,
    orientation: str,
    only_alive: bool,
    include_game_id: Union[bool, str],
    cache: bool = False,
    cache_dir: Optional[str] = None,
    metadata_df: Optional[pl.DataFrame] = None,
    teams_df: Optional[pl.DataFrame] = None,
    players_df: Optional[pl.DataFrame] = None,
    periods_df: Optional[pl.DataFrame] = None,
    **provider_kwargs,
) -> Union[pl.LazyFrame, CacheLazyResult]:
    """Create a lazy tracking DataFrame using register_io_source.

    This returns a REAL pl.LazyFrame with full Polars functionality.
    Data is only loaded when .collect() is called.

    Parameters
    ----------
    provider : str
        Provider name (e.g., "secondspectrum", "skillcorner")
    raw_data : FileLike
        Raw tracking data file
    meta_data : FileLike
        Metadata file
    schema : dict
        Schema mapping column names to Polars types
    layout : str
        DataFrame layout ("long", "long_ball", "wide")
    coordinates : str
        Coordinate system
    orientation : str
        Coordinate orientation
    only_alive : bool
        Filter to only alive frames
    include_game_id : bool or str
        Whether to include game_id column
    cache : bool
        If True, cache parsed data as Parquet for faster subsequent loads
    cache_dir : str, optional
        Cache directory path or URI (e.g., "s3://bucket/cache").
        If None, uses platform-specific default cache directory.
    metadata_df : pl.DataFrame, optional
        Metadata DataFrame to cache (required when cache=True)
    teams_df : pl.DataFrame, optional
        Teams DataFrame to cache (required when cache=True)
    players_df : pl.DataFrame, optional
        Players DataFrame to cache (required when cache=True)
    periods_df : pl.DataFrame, optional
        Periods DataFrame to cache (required when cache=True)
    **provider_kwargs
        Provider-specific parameters

    Returns
    -------
    pl.LazyFrame or tuple
        If cache hit with metadata: tuple of (LazyFrame, metadata_df, teams_df, players_df, periods_df)
        Otherwise: just the LazyFrame
    """
    from kloppy_light._base import get_provider

    # Build config string including provider-specific params
    config_parts = [
        layout,
        coordinates,
        orientation,
        str(only_alive),
        str(include_game_id),  # Include full value, not just bool (custom strings matter)
    ]
    # Add provider-specific params in sorted order for determinism
    for key in sorted(provider_kwargs.keys()):
        config_parts.append(f"{key}={provider_kwargs[key]}")
    config_str = "|".join(config_parts)

    # If caching is enabled, check for cache hit first
    if cache:
        from kloppy_light._cache import (
            cache_exists,
            compute_cache_key,
            compute_cache_key_fast,
            get_cache_path,
            read_cache,
            write_cache,
        )

        # Check if both files are local (can use fast cache key)
        is_local = _is_local_file(raw_data) and _is_local_file(meta_data)

        if is_local:
            # Fast path: compute key from path + mtime + size (no file reads)
            cache_key = compute_cache_key_fast(
                str(raw_data),
                str(meta_data),
                config_str,
            )
            cache_path = get_cache_path(cache_key, provider, cache_dir)

            if cache_exists(cache_path):
                # Cache hit! Return lazy reader (scan_parquet for local)
                logger.info(f"Cache hit for {provider}: loading from {cache_path}")
                return read_cache(cache_path)

            # Cache miss - files will be read in source_generator
            _cached_raw_bytes = None
            _cached_meta_bytes = None
        else:
            # Remote files: must read to compute content hash
            with open_as_file(raw_data) as f:
                raw_bytes = f.read() if f else b""
            with open_as_file(meta_data) as f:
                meta_bytes = f.read() if f else b""

            cache_key = compute_cache_key(
                raw_bytes,
                meta_bytes,
                config_str,
            )
            cache_path = get_cache_path(cache_key, provider, cache_dir)

            if cache_exists(cache_path):
                # Cache hit! Return lazy reader
                logger.info(f"Cache hit for {provider}: loading from {cache_path}")
                return read_cache(cache_path)

            # Cache miss - store bytes for reuse (avoid reading files twice)
            _cached_raw_bytes = raw_bytes
            _cached_meta_bytes = meta_bytes
    else:
        cache_path = None
        _cached_raw_bytes = None
        _cached_meta_bytes = None

    # Get provider config eagerly (small, fast)
    config = get_provider(provider)
    rust_module = config["rust_module"]
    tracking_params = config.get("tracking_params", [])

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        """Generator that yields tracking DataFrame.

        This is called by Polars when .collect() is invoked.
        The predicate and with_columns are passed for optimization.
        """
        # Use cached bytes if available (from cache check), otherwise read files
        if _cached_raw_bytes is not None:
            raw_bytes = _cached_raw_bytes
            meta_bytes = _cached_meta_bytes
        else:
            # Read files LAZILY here, at .collect() time
            with open_as_file(raw_data) as f:
                raw_bytes = f.read() if f else b""
            with open_as_file(meta_data) as f:
                meta_bytes = f.read() if f else b""

        # Build kwargs for Rust call
        call_kwargs = {
            "layout": layout,
            "coordinates": coordinates,
            "orientation": orientation,
            "only_alive": only_alive,
            "include_game_id": include_game_id,
        }

        # Add provider-specific params
        for param_name in tracking_params:
            if param_name in provider_kwargs:
                call_kwargs[param_name] = provider_kwargs[param_name]

        # Call Rust to load tracking data
        # When caching, we load WITHOUT predicate to cache full data
        # Then apply predicate after (for both cached and non-cached paths)
        if cache and cache_path:
            # For caching, load full data without predicate
            tracking_df, _, _, _, _ = rust_module.load_tracking(
                raw_bytes, meta_bytes, **call_kwargs
            )

            # Populate missing players before caching
            cache_players_df = players_df
            if players_df is not None and players_df.height == 0 and "player_id" in tracking_df.columns:
                from kloppy_light._dataset import extract_players_from_tracking
                cache_players_df = extract_players_from_tracking(
                    tracking_df,
                    periods_df,
                    existing_players_df=players_df,
                )

            # Write to cache with metadata (non-fatal if it fails)
            try:
                write_cache(
                    tracking_df,
                    cache_path,
                    metadata_df=metadata_df,
                    teams_df=teams_df,
                    players_df=cache_players_df,
                    periods_df=periods_df,
                )
            except Exception:
                pass  # Cache write failure is non-fatal

            # Apply predicate after caching full data
            if predicate is not None:
                tracking_df = tracking_df.filter(predicate)
        else:
            # No caching - use predicate pushdown for efficiency
            try:
                tracking_df, _, _, _, _ = rust_module.load_tracking(
                    raw_bytes, meta_bytes, predicate=predicate, **call_kwargs
                )
            except RuntimeError as e:
                if "deserializing" in str(e) or "BindingsError" in str(e):
                    # Fall back to loading without predicate
                    tracking_df, _, _, _, _ = rust_module.load_tracking(
                        raw_bytes, meta_bytes, **call_kwargs
                    )
                else:
                    raise

            # Always apply predicate in Python to handle filters Rust couldn't push down
            # This ensures correctness for team_id, player_id, is_between, etc.
            if predicate is not None:
                tracking_df = tracking_df.filter(predicate)

        # Apply column projection
        if with_columns is not None:
            tracking_df = tracking_df.select(with_columns)

        # Apply row limit
        if n_rows is not None:
            tracking_df = tracking_df.head(n_rows)

        yield tracking_df

    return pl.io.plugins.register_io_source(
        io_source=source_generator,
        schema=schema,
    )


def create_lazy_tracking_hawkeye(
    ball_data: Union[FileLike, List[FileLike]],
    player_data: Union[FileLike, List[FileLike]],
    meta_data: FileLike,
    schema: Dict[str, pl.DataType],
    layout: str,
    coordinates: str,
    orientation: str,
    only_alive: bool,
    pitch_length: float,
    pitch_width: float,
    object_id: str,
    include_game_id: Union[bool, str],
    cache: bool = False,
    cache_dir: Optional[str] = None,
    metadata_df: Optional[pl.DataFrame] = None,
    teams_df: Optional[pl.DataFrame] = None,
    players_df: Optional[pl.DataFrame] = None,
    periods_df: Optional[pl.DataFrame] = None,
) -> Union[pl.LazyFrame, CacheLazyResult]:
    """Create a lazy tracking DataFrame for HawkEye's dual-input format.

    HawkEye requires separate ball and player files, so it needs special handling.

    Parameters
    ----------
    ball_data : FileLike or List[FileLike]
        Ball tracking file(s)
    player_data : FileLike or List[FileLike]
        Player tracking file(s)
    meta_data : FileLike
        Metadata file
    schema : dict
        Schema mapping column names to Polars types
    layout : str
        DataFrame layout
    coordinates : str
        Coordinate system
    orientation : str
        Coordinate orientation
    only_alive : bool
        Filter to only alive frames
    pitch_length : float
        Pitch length in meters
    pitch_width : float
        Pitch width in meters
    object_id : str
        Object ID preference
    include_game_id : bool or str
        Whether to include game_id column
    cache : bool
        If True, cache parsed data as Parquet for faster subsequent loads
    cache_dir : str, optional
        Cache directory path or URI (e.g., "s3://bucket/cache").
    metadata_df : pl.DataFrame, optional
        Metadata DataFrame to cache (required when cache=True)
    teams_df : pl.DataFrame, optional
        Teams DataFrame to cache (required when cache=True)
    players_df : pl.DataFrame, optional
        Players DataFrame to cache (required when cache=True)
    periods_df : pl.DataFrame, optional
        Periods DataFrame to cache (required when cache=True)

    Returns
    -------
    pl.LazyFrame or tuple
        If cache hit with metadata: tuple of (LazyFrame, metadata_df, teams_df, players_df, periods_df)
        Otherwise: just the LazyFrame
    """
    import hashlib

    from kloppy_light._base import get_filename_from_filelike
    from kloppy_light._kloppy_light import hawkeye as _hawkeye

    # Only convert to lists eagerly (no file reading)
    ball_list = ball_data if isinstance(ball_data, list) else [ball_data]
    player_list = player_data if isinstance(player_data, list) else [player_data]

    # If caching is enabled, check for cache hit first
    if cache:
        from kloppy_light._cache import (
            CACHE_SCHEMA_VERSION,
            cache_exists,
            compute_cache_key_fast_multi,
            get_cache_path,
            read_cache,
            write_cache,
        )

        # Check if all files are local (can use fast cache key)
        all_files = list(ball_list) + list(player_list) + [meta_data]
        is_local = all(_is_local_file(f) for f in all_files)

        if is_local:
            # Fast path: compute key from path + mtime + size (no file reads)
            all_paths = [str(f) for f in ball_list] + [str(f) for f in player_list]
            config_str = (
                f"{layout}|{coordinates}|{orientation}|{only_alive}|"
                f"{pitch_length}|{pitch_width}|{object_id}|{include_game_id}"
            )
            cache_key = compute_cache_key_fast_multi(
                all_paths, str(meta_data), config_str
            )
            cache_path = get_cache_path(cache_key, "hawkeye", cache_dir)

            if cache_exists(cache_path):
                logger.info(f"Cache hit for hawkeye: loading from {cache_path}")
                return read_cache(cache_path)

            # Cache miss - files will be read in source_generator
            _cached_ball_bytes = None
            _cached_player_bytes = None
            _cached_meta_bytes = None
        else:
            # Remote files: must read to compute content hash
            ball_bytes_list: List[Tuple[str, bytes]] = []
            for ball_file in ball_list:
                with open_as_file(ball_file) as f:
                    filename = get_filename_from_filelike(ball_file)
                    ball_bytes_list.append((filename, f.read() if f else b""))

            player_bytes_list: List[Tuple[str, bytes]] = []
            for player_file in player_list:
                with open_as_file(player_file) as f:
                    filename = get_filename_from_filelike(player_file)
                    player_bytes_list.append((filename, f.read() if f else b""))

            with open_as_file(meta_data) as f:
                meta_bytes = f.read() if f else b""

            # Compute cache key from all file contents
            hasher = hashlib.sha256()
            for _, ball_bytes in ball_bytes_list:
                hasher.update(ball_bytes)
            for _, player_bytes in player_bytes_list:
                hasher.update(player_bytes)
            hasher.update(meta_bytes)
            hasher.update(
                f"{layout}|{coordinates}|{orientation}|{only_alive}|"
                f"{pitch_length}|{pitch_width}|{object_id}|{include_game_id}|"
                f"{CACHE_SCHEMA_VERSION}".encode()
            )
            cache_key = hasher.hexdigest()[:16]
            cache_path = get_cache_path(cache_key, "hawkeye", cache_dir)

            if cache_exists(cache_path):
                logger.info(f"Cache hit for hawkeye: loading from {cache_path}")
                return read_cache(cache_path)

            # Store for reuse in source_generator
            _cached_ball_bytes = ball_bytes_list
            _cached_player_bytes = player_bytes_list
            _cached_meta_bytes = meta_bytes
    else:
        cache_path = None
        _cached_ball_bytes = None
        _cached_player_bytes = None
        _cached_meta_bytes = None

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        """Generator that yields HawkEye tracking DataFrame."""
        # Use cached bytes if available, otherwise read files
        if _cached_ball_bytes is not None:
            ball_bytes_list = _cached_ball_bytes
            player_bytes_list = _cached_player_bytes
            meta_bytes = _cached_meta_bytes
        else:
            # Read files LAZILY here, at .collect() time
            ball_bytes_list_inner: List[Tuple[str, bytes]] = []
            for ball_file in ball_list:
                with open_as_file(ball_file) as f:
                    filename = get_filename_from_filelike(ball_file)
                    ball_bytes_list_inner.append((filename, f.read() if f else b""))
            ball_bytes_list = ball_bytes_list_inner

            player_bytes_list_inner: List[Tuple[str, bytes]] = []
            for player_file in player_list:
                with open_as_file(player_file) as f:
                    filename = get_filename_from_filelike(player_file)
                    player_bytes_list_inner.append((filename, f.read() if f else b""))
            player_bytes_list = player_bytes_list_inner

            with open_as_file(meta_data) as f:
                meta_bytes = f.read() if f else b""

        # Call Rust to load tracking data
        if cache and cache_path:
            # For caching, load full data without predicate
            tracking_df, _, _, _, _ = _hawkeye.load_tracking(
                ball_bytes_list,
                player_bytes_list,
                meta_bytes,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                object_id=object_id,
                include_game_id=include_game_id,
            )

            # Populate missing players before caching
            cache_players_df = players_df
            if players_df is not None and players_df.height == 0 and "player_id" in tracking_df.columns:
                from kloppy_light._dataset import extract_players_from_tracking
                cache_players_df = extract_players_from_tracking(
                    tracking_df,
                    periods_df,
                    existing_players_df=players_df,
                )

            # Write to cache with metadata
            try:
                write_cache(
                    tracking_df,
                    cache_path,
                    metadata_df=metadata_df,
                    teams_df=teams_df,
                    players_df=cache_players_df,
                    periods_df=periods_df,
                )
            except Exception:
                pass

            if predicate is not None:
                tracking_df = tracking_df.filter(predicate)
        else:
            # No caching - use predicate pushdown
            try:
                tracking_df, _, _, _, _ = _hawkeye.load_tracking(
                    ball_bytes_list,
                    player_bytes_list,
                    meta_bytes,
                    layout=layout,
                    coordinates=coordinates,
                    orientation=orientation,
                    only_alive=only_alive,
                    pitch_length=pitch_length,
                    pitch_width=pitch_width,
                    object_id=object_id,
                    include_game_id=include_game_id,
                    predicate=predicate,
                )
            except RuntimeError as e:
                if "deserializing" in str(e) or "BindingsError" in str(e):
                    # Fall back to loading without predicate
                    tracking_df, _, _, _, _ = _hawkeye.load_tracking(
                        ball_bytes_list,
                        player_bytes_list,
                        meta_bytes,
                        layout=layout,
                        coordinates=coordinates,
                        orientation=orientation,
                        only_alive=only_alive,
                        pitch_length=pitch_length,
                        pitch_width=pitch_width,
                        object_id=object_id,
                        include_game_id=include_game_id,
                    )
                else:
                    raise

            # Always apply predicate in Python to handle filters Rust couldn't push down
            # This ensures correctness for team_id, player_id, is_between, etc.
            if predicate is not None:
                tracking_df = tracking_df.filter(predicate)

        # Apply column projection
        if with_columns is not None:
            tracking_df = tracking_df.select(with_columns)

        # Apply row limit
        if n_rows is not None:
            tracking_df = tracking_df.head(n_rows)

        yield tracking_df

    return pl.io.plugins.register_io_source(
        io_source=source_generator,
        schema=schema,
    )
