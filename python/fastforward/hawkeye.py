"""HawkEye provider wrapper.

HawkEye tracking data consists of multiple per-minute files:
- Ball files: hawkeye_{period_id}_{minute}.football.samples.ball
- Player files: hawkeye_{period_id}_{minute}.football.samples.centroids

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
from fastforward._errors import with_error_handler
from fastforward._fastforward import hawkeye as _hawkeye
from fastforward._lazy import create_lazy_tracking_hawkeye, _is_local_file
from fastforward._schema import get_tracking_schema
from fastforward._dataset import TrackingDataset

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


@with_error_handler
def load_tracking(
    ball_data: Union[FileLike, List[FileLike]],
    player_data: Union[FileLike, List[FileLike]],
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "hawkeye",
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
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["fifa", "uefa", "he", "auto"] = "auto",
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
    Load HawkEye tracking data.

    Parameters
    ----------
    ball_data : FileLike or List[FileLike]
        Ball tracking file(s). Can be:
        - Single FileLike: Path, bytes, or file-like object
        - List[FileLike]: Multiple ball files (one per minute)
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths, zip files.
    player_data : FileLike or List[FileLike]
        Player tracking file(s). Can be:
        - Single FileLike: Path, bytes, or file-like object
        - List[FileLike]: Multiple player files (one per minute)
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths, zip files.
    meta_data : FileLike
        Path to metadata file (JSON or XML), or bytes, or file-like object.
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths, zip files.
    layout : {"long"}, default "long"
        DataFrame layout. Currently only "long" is supported.
        - "long": Ball as row with team_id="ball", player_id="ball"
        (TODO: "long_ball" and "wide" layouts)
    coordinates : {"cdf"}, default "cdf"
        Coordinate system. Currently only "cdf" is supported.
        - "cdf": Common Data Format (origin at center, meters)
        (TODO: Other coordinate systems)
    orientation : {"static_home_away"}, default "static_home_away"
        Coordinate orientation. Currently only "static_home_away" is supported.
        - "static_home_away": Home attacks right (+x) entire match
        (TODO: Other orientations)
    only_alive : bool, default True
        If True, only include frames where ball is in play (play field == "In").
        Uses HawkEye's "play" field instead of typical "live" field.
    pitch_length : float, default 105.0
        Pitch length in meters. Used as fallback if not in metadata.
        Metadata values take precedence if present.
    pitch_width : float, default 68.0
        Pitch width in meters. Used as fallback if not in metadata.
        Metadata values take precedence if present.
    object_id : {"fifa", "uefa", "he", "auto"} or str, default "auto"
        Object ID preference for team and player identification:
        - "fifa": Use FIFA IDs (error if not present)
        - "uefa": Use UEFA IDs (error if not present)
        - "he": Use HawkEye IDs
        - "auto": Prefer FIFA > UEFA > HawkEye (automatic fallback)
        - Custom string: Use custom ID field (error if not found)
    include_game_id : bool or str, default True
        If True, add game_id column to tracking_df, team_df, and player_df from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_officials : bool, default False
        If True, include officials in player_df and tracking data with team_id="officials"
        and position codes: REF (Main Referee), AREF (Assistant Referee).
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
        If engine="polars", .tracking returns pl.DataFrame.
        If engine="pyspark", all DataFrames are PySpark DataFrames.

    Notes
    -----
    - Officials are excluded by default; set include_officials=True to include them
    - Period and minute are extracted from filename patterns like hawkeye_1_1.ball

    Examples
    --------
    Load from file paths:

    >>> ball_files = ["hawkeye_1_1.ball", "hawkeye_1_2.ball"]
    >>> player_files = ["hawkeye_1_1.centroids", "hawkeye_1_2.centroids"]
    >>> dataset = load_tracking(ball_files, player_files, "hawkeye_meta.json")
    >>> tracking_df = dataset.tracking

    Load with specific object ID preference:

    >>> dataset = load_tracking(ball_files, player_files, "hawkeye_meta.json", object_id="fifa")

    PySpark engine:

    >>> dataset = load_tracking(ball_files, player_files, "hawkeye_meta.json", engine="pyspark")
    >>> dataset.tracking.show(5)
    """
    from fastforward._engine import validate_engine, polars_to_spark, get_spark_session

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
        # Handle directory input for ball_data
        if isinstance(ball_data, (str, Path)) and Path(ball_data).is_dir():
            ball_data_processed = discover_files_in_directory(ball_data, "*.ball")
        elif isinstance(ball_data, list):
            ball_data_processed = ball_data
        else:
            ball_data_processed = [ball_data]

        # Handle directory input for player_data
        if isinstance(player_data, (str, Path)) and Path(player_data).is_dir():
            player_data_processed = discover_files_in_directory(player_data, "*.centroids")
        elif isinstance(player_data, list):
            player_data_processed = player_data
        else:
            player_data_processed = [player_data]

        # Validate counts match
        if len(ball_data_processed) != len(player_data_processed):
            raise ValueError(
                f"Mismatch: {len(ball_data_processed)} ball files but "
                f"{len(player_data_processed)} player files"
            )

        # Build config string for cache key
        config_str = (
            f"{layout}|{coordinates}|{orientation}|{only_alive}|"
            f"{pitch_length}|{pitch_width}|{object_id}|{include_game_id}|{include_officials}"
        )

        # Compute cache key
        cache_key: Optional[str] = None
        all_files = list(ball_data_processed) + list(player_data_processed) + [meta_data]
        is_local = all(_is_local_file(f) for f in all_files)

        if is_local:
            from fastforward._cache import compute_cache_key_fast_multi

            all_paths = [str(f) for f in ball_data_processed] + [str(f) for f in player_data_processed]
            try:
                cache_key = compute_cache_key_fast_multi(
                    all_paths, str(meta_data), config_str
                )
            except FileNotFoundError:
                # Files don't exist, cache key cannot be computed
                cache_key = None

        # Check for cache hit if from_cache=True
        if from_cache and cache_key:
            from fastforward._cache import cache_exists, get_cache_path, read_cache

            cache_path = get_cache_path(cache_key, "hawkeye")
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
                        _provider="hawkeye",
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

        # Load metadata only (fast) - NOW WITH TEAM/PLAYER DATA
        metadata_df, team_df, player_df, periods_df = load_metadata_only(
            meta_data,
            player_data=player_data_processed,  # Pass player data for teams and players
            coordinates=coordinates,
            orientation=orientation,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            object_id=object_id,
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
        lazy_frame = create_lazy_tracking_hawkeye(
            ball_data=ball_data_processed,
            player_data=player_data_processed,
            meta_data=meta_data,
            schema=schema,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            object_id=object_id,
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
            _provider="hawkeye",
            _cache_key=cache_key,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )

    # Eager loading (existing logic)
    # Convert FileLike to bytes for metadata
    with open_as_file(meta_data) as meta_file:
        meta_bytes = meta_file.read() if meta_file else b""

    # Handle directory input for ball_data
    if isinstance(ball_data, (str, Path)) and Path(ball_data).is_dir():
        ball_data_list = discover_files_in_directory(ball_data, "*.ball")
    elif isinstance(ball_data, list):
        ball_data_list = ball_data
    else:
        ball_data_list = [ball_data]

    # Handle directory input for player_data
    if isinstance(player_data, (str, Path)) and Path(player_data).is_dir():
        player_data_list = discover_files_in_directory(player_data, "*.centroids")
    elif isinstance(player_data, list):
        player_data_list = player_data
    else:
        player_data_list = [player_data]

    # Validate counts match
    if len(ball_data_list) != len(player_data_list):
        raise ValueError(
            f"Mismatch: {len(ball_data_list)} ball files but "
            f"{len(player_data_list)} player files"
        )

    # Convert ball_data to list of (filename, bytes) tuples
    ball_bytes_list = []
    for ball_file in ball_data_list:
        filename = get_filename_from_filelike(ball_file)
        with open_as_file(ball_file) as f:
            ball_bytes_list.append((filename, f.read() if f else b""))

    # Convert player_data to list of (filename, bytes) tuples
    player_bytes_list = []
    for player_file in player_data_list:
        filename = get_filename_from_filelike(player_file)
        with open_as_file(player_file) as f:
            player_bytes_list.append((filename, f.read() if f else b""))

    # Pass bytes to Rust
    tracking_df, metadata_df, team_df, player_df, periods_df = _hawkeye.load_tracking(
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
        include_officials=include_officials,
        parallel=parallel,
    )

    # Compute cache key for eager loading too
    cache_key = None
    all_files = list(ball_data_list) + list(player_data_list) + [meta_data]
    is_local = all(_is_local_file(f) for f in all_files)
    if is_local:
        from fastforward._cache import compute_cache_key_fast_multi

        config_str = (
            f"{layout}|{coordinates}|{orientation}|{only_alive}|"
            f"{pitch_length}|{pitch_width}|{object_id}|{include_game_id}|{include_officials}"
        )
        all_paths = [str(f) for f in ball_data_list] + [str(f) for f in player_data_list]
        cache_key = compute_cache_key_fast_multi(
            all_paths, str(meta_data), config_str
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
            _provider="hawkeye",
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
        _provider="hawkeye",
        _cache_key=cache_key,
        _coordinate_system=coordinates,
        _orientation=orientation,
    )


@with_error_handler
def load_metadata_only(
    meta_data: FileLike,
    player_data: Optional[Union[FileLike, List[FileLike]]] = None,
    coordinates: Literal[
        "cdf",
        "hawkeye",
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
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["auto", "heId", "fifaId"] = "auto",
    include_game_id: Union[bool, str] = True,
    include_officials: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load only HawkEye metadata without tracking data.

    Parameters
    ----------
    meta_data : FileLike
        Path to metadata file (JSON or XML), or bytes, or file-like object.
    player_data : FileLike or List[FileLike], optional
        Optional path(s) to player centroid file(s) for team and player extraction.
        Only the first file is used as it contains all teams and players.
        If provided, team_df and player_df will be populated.
    coordinates : {"cdf"}, default "cdf"
        Coordinate system (currently only "cdf" supported).
    orientation : {"static_home_away"}, default "static_home_away"
        Coordinate orientation (currently only "static_home_away" supported).
    pitch_length : float, default 105.0
        Pitch length in meters (fallback if not in metadata).
    pitch_width : float, default 68.0
        Pitch width in meters (fallback if not in metadata).
    object_id : {"auto", "heId", "fifaId"}, default "auto"
        Which ID system to use for teams/players. "auto" tries heId first, falls back to fifaId.
    include_game_id : bool or str, default True
        If True, add game_id column from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_officials : bool, default False
        If True, include officials in player_df with team_id="officials".

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (metadata_df, team_df, player_df, periods_df)

    Notes
    -----
    HawkEye metadata files (JSON/XML) only contain match info and pitch dimensions.
    To get team and player information, pass player_data. The first player centroid file
    contains all players from both teams, so only one file is needed.
    """
    # Convert FileLike to bytes
    with open_as_file(meta_data) as meta_file:
        meta_bytes = meta_file.read() if meta_file else b""

    # Read first player file (contains all teams and players)
    player_bytes = None
    if player_data is not None:
        # Handle directory input for player_data
        if isinstance(player_data, (str, Path)) and Path(player_data).is_dir():
            player_list = discover_files_in_directory(player_data, "*.centroids")
        elif isinstance(player_data, list):
            player_list = player_data
        else:
            player_list = [player_data]

        # Only read first file - it contains ALL teams and players
        if player_list:
            with open_as_file(player_list[0]) as player_file:
                player_bytes = player_file.read() if player_file else None

    # Pass bytes to Rust
    return _hawkeye.load_metadata_only(
        meta_bytes,
        player_bytes,  # Pass single file, not a list
        coordinates=coordinates,
        orientation=orientation,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        object_id=object_id,
        include_game_id=include_game_id,
        include_officials=include_officials,
    )
