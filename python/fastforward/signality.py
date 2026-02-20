"""Signality provider wrapper.

Signality tracking data consists of:
- Metadata file: signality_meta_data.json (teams, players, lineups)
- Venue file: signality_venue_information.json (pitch dimensions)
- Raw data feeds: signality_p{period}_raw_data.json (per-period tracking files)

Supports both eager and lazy loading modes.
"""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import polars as pl
from kloppy.io import FileLike, open_as_file

from fastforward._base import (
    discover_files_in_directory,
    get_filename_from_filelike,
)
from fastforward._fastforward import signality as _signality
from fastforward._lazy import create_lazy_tracking_signality, _is_local_file
from fastforward._schema import get_tracking_schema
from fastforward._dataset import TrackingDataset

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def load_tracking(
    meta_data: FileLike,
    raw_data_feeds: Union[FileLike, List[FileLike]],
    venue_information: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "signality",
        "kloppy",
        "opta",
        "pff",
        "secondspectrum",
        "skillcorner",
        "sportec:event",
        "sportec:tracking",
        "sportvu",
        "tracab",
    ] = "cdf",
    orientation: Literal[
        "static_home_away",
        "attack_left",
        "attack_right",
        "away_home",
        "home_away",
        "static_away_home",
    ] = "static_home_away",
    only_alive: bool = True,
    include_game_id: Union[bool, str] = True,
    include_officials: bool = False,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    parallel: bool = True,
    engine: Literal["polars", "pyspark"] = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """
    Load Signality tracking data.

    Parameters
    ----------
    meta_data : FileLike
        Path to metadata file (JSON), or bytes, or file-like object.
        Contains team names, player info, lineups, and match timestamp.
    raw_data_feeds : FileLike or List[FileLike]
        Raw tracking data file(s). Can be:
        - Single FileLike: Path, bytes, or file-like object
        - List[FileLike]: Multiple files (one per period)
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths.
    venue_information : FileLike
        Path to venue information file (JSON) containing pitch dimensions.
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
    coordinates : str, default "cdf"
        Coordinate system. Options:
        - "cdf": Common Data Format (origin at center, meters)
        - "signality": Native coordinates (same as CDF)
        - Other provider coordinate systems
    orientation : str, default "static_home_away"
        Coordinate orientation:
        - "static_home_away": Home attacks right (+x) entire match
        - Other orientations available
    only_alive : bool, default True
        If True, only include frames where ball is in play ("running" state).
    include_game_id : bool or str, default True
        If True, add game_id column from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_officials : bool, default False
        If True, include officials in player_df and tracking data with team_id="officials"
        and position codes: REF (Main Referee), AREF (Assistant Referee), FOURTH (4th Official).
    parallel : bool, default True
        If True, process multiple files in parallel using rayon.
    engine : {"polars", "pyspark"}, default "polars"
        DataFrame engine to use:
        - "polars": Return Polars DataFrames (default)
        - "pyspark": Return PySpark DataFrames
    spark_session : SparkSession, optional
        PySpark SparkSession to use. If None and engine="pyspark",
        will get or create a session automatically.

    Returns
    -------
    TrackingDataset
        Object with .tracking, .metadata, .teams, .players, .periods properties.

    Notes
    -----
    - Signality uses center-origin coordinates in meters (same as CDF)
    - Period is extracted from filename patterns like signality_p1_raw_data.json
    - Frame rate is typically 25 Hz (40ms between frames)

    Examples
    --------
    Load from file paths:

    >>> from fastforward import signality
    >>> dataset = signality.load_tracking(
    ...     meta_data="signality_meta_data.json",
    ...     raw_data_feeds=["signality_p1_raw_data.json", "signality_p2_raw_data.json"],
    ...     venue_information="signality_venue_information.json",
    ... )
    >>> tracking_df = dataset.tracking
    """
    from fastforward._engine import (
        validate_engine,
        polars_to_spark,
        get_spark_session,
    )

    # Validate engine parameter
    engine = validate_engine(engine)

    if lazy:
        raise NotImplementedError("lazy loading is not yet supported in fast-forward")
    if from_cache:
        raise NotImplementedError("cache loading is not yet supported in fast-forward")

    # Wide format doesn't support lazy loading - column names are game-specific
    if lazy and layout == "wide":
        raise ValueError(
            "lazy=True is not supported for layout='wide'. "
            "Wide format has game-specific column names (player IDs), "
            "making lazy frame operations like concatenation incompatible."
        )

    # For PySpark, force eager loading (will convert after)
    if engine == "pyspark":
        lazy = False

    # Handle lazy loading
    if lazy:
        # Handle directory input for raw_data_feeds
        if isinstance(raw_data_feeds, (str, Path)) and Path(raw_data_feeds).is_dir():
            raw_data_processed = discover_files_in_directory(
                raw_data_feeds, "*raw_data*.json"
            )
        elif isinstance(raw_data_feeds, list):
            raw_data_processed = raw_data_feeds
        else:
            raw_data_processed = [raw_data_feeds]

        # Build config string for cache key
        config_str = (
            f"{layout}|{coordinates}|{orientation}|{only_alive}|"
            f"{include_game_id}|{include_officials}"
        )

        # Compute cache key
        cache_key: Optional[str] = None
        all_files = list(raw_data_processed) + [meta_data, venue_information]
        is_local = all(_is_local_file(f) for f in all_files)

        if is_local:
            from fastforward._cache import compute_cache_key_fast_multi

            # Include venue_information in file_paths since function only accepts single meta_path
            all_paths = [str(f) for f in raw_data_processed] + [str(venue_information)]
            try:
                cache_key = compute_cache_key_fast_multi(
                    all_paths,
                    str(meta_data),
                    config_str,
                )
            except FileNotFoundError:
                # Files don't exist, cache key cannot be computed
                cache_key = None

        # Check for cache hit if from_cache=True
        if from_cache and cache_key:
            from fastforward._cache import cache_exists, get_cache_path, read_cache

            cache_path = get_cache_path(cache_key, "signality")
            if cache_exists(cache_path):
                # Cache hit - load from cache
                result = read_cache(cache_path)
                if isinstance(result, tuple):
                    lazy_frame, metadata_df, team_df, player_df, periods_df = result
                    return TrackingDataset(
                        tracking=lazy_frame,
                        metadata=metadata_df,
                        teams=team_df,
                        players=player_df,
                        periods=periods_df,
                        _engine="polars",
                        _provider="signality",
                        _cache_key=cache_key,
                        _coordinate_system=coordinates,
                        _orientation=orientation,
                    )
            else:
                # Cache miss with from_cache=True - warn user
                warnings.warn(
                    "No cache found for this file. "
                    "Use dataset.write_cache() after loading to create one.",
                    UserWarning,
                )

        # Load metadata only (fast)
        metadata_df, team_df, player_df, periods_df = load_metadata_only(
            meta_data,
            venue_information,
            coordinates=coordinates,
            orientation=orientation,
            include_game_id=include_game_id,
            include_officials=include_officials,
        )

        # Generate schema for the tracking DataFrame
        schema = get_tracking_schema(
            layout=layout,
            players_df=player_df,
            include_game_id=bool(include_game_id),
        )

        # Create real pl.LazyFrame using register_io_source
        lazy_frame = create_lazy_tracking_signality(
            raw_data_feeds=raw_data_processed,
            meta_data=meta_data,
            venue_information=venue_information,
            schema=schema,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            include_game_id=include_game_id,
            include_officials=include_officials,
            parallel=parallel,
        )

        return TrackingDataset(
            tracking=lazy_frame,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
            _engine="polars",
            _provider="signality",
            _cache_key=cache_key,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )

    # Eager loading
    # Convert FileLike to bytes for metadata and venue
    with open_as_file(meta_data) as meta_file:
        meta_bytes = meta_file.read() if meta_file else b""
    with open_as_file(venue_information) as venue_file:
        venue_bytes = venue_file.read() if venue_file else b""

    # Handle directory input for raw_data_feeds
    if isinstance(raw_data_feeds, (str, Path)) and Path(raw_data_feeds).is_dir():
        raw_data_list = discover_files_in_directory(
            raw_data_feeds, "*raw_data*.json"
        )
    elif isinstance(raw_data_feeds, list):
        raw_data_list = raw_data_feeds
    else:
        raw_data_list = [raw_data_feeds]

    # Convert raw_data to list of (filename, bytes) tuples
    raw_bytes_list = []
    for raw_file in raw_data_list:
        filename = get_filename_from_filelike(raw_file)
        with open_as_file(raw_file) as f:
            raw_bytes_list.append((filename, f.read() if f else b""))

    # Pass bytes to Rust
    tracking_df, metadata_df, team_df, player_df, periods_df = _signality.load_tracking(
        raw_bytes_list,
        meta_bytes,
        venue_bytes,
        layout=layout,
        coordinates=coordinates,
        orientation=orientation,
        only_alive=only_alive,
        include_game_id=include_game_id,
        include_officials=include_officials,
        parallel=parallel,
    )

    # Compute cache key for eager loading too
    cache_key = None
    all_files = list(raw_data_list) + [meta_data, venue_information]
    is_local = all(_is_local_file(f) for f in all_files)
    if is_local:
        from fastforward._cache import compute_cache_key_fast_multi

        config_str = (
            f"{layout}|{coordinates}|{orientation}|{only_alive}|"
            f"{include_game_id}|{include_officials}"
        )
        # Include venue_information in file_paths since function only accepts single meta_path
        all_paths = [str(f) for f in raw_data_list] + [str(venue_information)]
        cache_key = compute_cache_key_fast_multi(
            all_paths,
            str(meta_data),
            config_str,
        )

    # Convert to PySpark if requested
    if engine == "pyspark":
        spark = spark_session or get_spark_session()
        return TrackingDataset(
            tracking=polars_to_spark(tracking_df, spark),
            metadata=polars_to_spark(metadata_df, spark),
            teams=polars_to_spark(team_df, spark),
            players=polars_to_spark(player_df, spark),
            periods=polars_to_spark(periods_df, spark),
            _engine="pyspark",
            _provider="signality",
            _cache_key=cache_key,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )

    return TrackingDataset(
        tracking=tracking_df,
        metadata=metadata_df,
        teams=team_df,
        players=player_df,
        periods=periods_df,
        _engine="polars",
        _provider="signality",
        _cache_key=cache_key,
        _coordinate_system=coordinates,
        _orientation=orientation,
    )


def load_metadata_only(
    meta_data: FileLike,
    venue_information: FileLike,
    coordinates: str = "cdf",
    orientation: str = "static_home_away",
    include_game_id: Union[bool, str] = True,
    include_officials: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load only Signality metadata without tracking data.

    Parameters
    ----------
    meta_data : FileLike
        Path to metadata file (JSON), or bytes, or file-like object.
    venue_information : FileLike
        Path to venue information file (JSON) containing pitch dimensions.
    coordinates : str, default "cdf"
        Coordinate system (metadata output).
    orientation : str, default "static_home_away"
        Coordinate orientation (metadata output).
    include_game_id : bool or str, default True
        If True, add game_id column from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_officials : bool, default False
        If True, include officials placeholder in player_df.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (metadata_df, team_df, player_df, periods_df)

    Notes
    -----
    Signality metadata contains team names, player info, and lineups.
    Venue information contains pitch dimensions.
    """
    # Convert FileLike to bytes
    with open_as_file(meta_data) as meta_file:
        meta_bytes = meta_file.read() if meta_file else b""
    with open_as_file(venue_information) as venue_file:
        venue_bytes = venue_file.read() if venue_file else b""

    # Pass bytes to Rust
    return _signality.load_metadata_only(
        meta_bytes,
        venue_bytes,
        coordinates=coordinates,
        orientation=orientation,
        include_game_id=include_game_id,
        include_officials=include_officials,
    )
