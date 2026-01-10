"""HawkEye provider wrapper.

HawkEye tracking data consists of multiple per-minute files:
- Ball files: hawkeye_{period_id}_{minute}.football.samples.ball
- Player files: hawkeye_{period_id}_{minute}.football.samples.centroids

Supports both eager and lazy loading modes.
"""

from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, overload
import polars as pl

from kloppy.io import FileLike, open_as_file

from kloppy_light._kloppy_light import hawkeye as _hawkeye
from kloppy_light._lazy import LazyTrackingLoader
from kloppy_light._dataset import TrackingDataset


def _get_filename(filelike: FileLike) -> str:
    """Extract filename from FileLike object.

    Parameters
    ----------
    filelike : FileLike
        FileLike object to extract filename from

    Returns
    -------
    str
        Filename (without path) or empty string if not extractable
    """
    if isinstance(filelike, str):
        return Path(filelike).name
    elif isinstance(filelike, Path):
        return filelike.name
    elif hasattr(filelike, 'name'):
        # File-like object with name attribute
        return Path(str(filelike.name)).name
    else:
        # Bytes or unknown type - no filename available
        return ""


def _discover_files_in_directory(
    directory: Union[str, Path],
    pattern: str
) -> List[Path]:
    """Discover files matching pattern in directory, sorted by period/minute.

    Parameters
    ----------
    directory : Union[str, Path]
        Directory path to search in
    pattern : str
        Glob pattern to match files (e.g., "*.ball", "*.centroids")

    Returns
    -------
    List[Path]
        Sorted list of matching file paths

    Raises
    ------
    ValueError
        If directory doesn't exist or no matching files found
    """
    import re

    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    files = list(dir_path.glob(pattern))

    if not files:
        raise ValueError(f"No files matching '{pattern}' found in {directory}")

    # Sort by (period, minute) extracted from filename
    def sort_key(path: Path) -> Tuple[int, int]:
        # Pattern: {prefix}_{period}_{minute}[_{extra_minute}].{extension}
        # Match the LAST 2-3 digit groups before the file extension (anchored to end)
        match = re.search(r'_(\d{1,2})_(\d{1,3})(?:_(\d{1,2}))?\.(?:football\.samples\.)?(ball|centroids)$', path.name)
        if match:
            period = int(match.group(1))
            base_minute = int(match.group(2))
            extra_minute = int(match.group(3)) if match.group(3) else 0
            total_minute = base_minute + extra_minute
            return (period, total_minute)
        return (999, 999)  # Unparseable files at end

    return sorted(files, key=sort_key)


@overload
def load_tracking(
    ball_data: Union[FileLike, List[FileLike]],
    player_data: Union[FileLike, List[FileLike]],
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal["static_home_away"] = "static_home_away",
    only_alive: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["fifa", "uefa", "he", "auto"] = "auto",
    include_game_id: Union[bool, str] = True,
    *,
    lazy: Literal[False],
) -> TrackingDataset: ...


@overload
def load_tracking(
    ball_data: Union[FileLike, List[FileLike]],
    player_data: Union[FileLike, List[FileLike]],
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal["static_home_away"] = "static_home_away",
    only_alive: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["fifa", "uefa", "he", "auto"] = "auto",
    include_game_id: Union[bool, str] = True,
    *,
    lazy: Literal[True] = True,
) -> TrackingDataset: ...


def load_tracking(
    ball_data: Union[FileLike, List[FileLike]],
    player_data: Union[FileLike, List[FileLike]],
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal["static_home_away"] = "static_home_away",
    only_alive: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["fifa", "uefa", "he", "auto"] = "auto",
    include_game_id: Union[bool, str] = True,
    *,
    lazy: bool = True,
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
    lazy : bool, default True
        If True, return a TrackingDataset with LazyTrackingLoader for tracking.
        If False, return a TrackingDataset with eager DataFrame for tracking.
        Lazy loading is useful for large datasets where you want to filter before loading all data.

    Returns
    -------
    TrackingDataset
        Object with .tracking, .metadata, .teams, .players, .periods properties.
        If lazy=True, .tracking returns LazyTrackingLoader (call .collect() to get DataFrame).
        If lazy=False, .tracking returns pl.DataFrame directly.

    Notes
    -----
    - Officials are automatically included in player_df if present in tracking data
    - Period and minute are extracted from filename patterns like hawkeye_1_1.ball
    - Lazy loading only parses tracking data when .collect() is called

    Examples
    --------
    Load from file paths:

    >>> ball_files = ["hawkeye_1_1.ball", "hawkeye_1_2.ball"]
    >>> player_files = ["hawkeye_1_1.centroids", "hawkeye_1_2.centroids"]
    >>> tracking_df, metadata_df, team_df, player_df = load_tracking(
    ...     ball_files, player_files, "hawkeye_meta.json"
    ... )

    Load with specific object ID preference:

    >>> tracking_df, metadata_df, team_df, player_df = load_tracking(
    ...     ball_files, player_files, "hawkeye_meta.json", object_id="fifa"
    ... )

    Lazy loading with filtering:
    >>> tracking_lazy, metadata_df, team_df, player_df = load_tracking(
    ...     ball_files, player_files, "hawkeye_meta.json", lazy=True
    ... )
    >>> # No data loaded yet
    >>> period1_df = tracking_lazy.filter(pl.col("period_id") == 1).collect()
    >>> # Now data is loaded and filtered
    """
    # Handle lazy loading
    if lazy:
        # Handle directory input for ball_data
        if isinstance(ball_data, (str, Path)) and Path(ball_data).is_dir():
            ball_data_processed = _discover_files_in_directory(ball_data, "*.ball")
        elif isinstance(ball_data, list):
            ball_data_processed = ball_data
        else:
            ball_data_processed = [ball_data]

        # Handle directory input for player_data
        if isinstance(player_data, (str, Path)) and Path(player_data).is_dir():
            player_data_processed = _discover_files_in_directory(player_data, "*.centroids")
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
        )

        # Create lazy loader with tuple of lists
        lazy_loader = LazyTrackingLoader(
            provider="hawkeye",
            raw_data=(ball_data_processed, player_data_processed),  # TUPLE of lists
            meta_data=meta_data,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            object_id=object_id,
            include_game_id=include_game_id,
        )

        return TrackingDataset(
            tracking=lazy_loader,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
        )

    # Eager loading (existing logic)
    # Convert FileLike to bytes for metadata
    with open_as_file(meta_data) as meta_file:
        meta_bytes = meta_file.read() if meta_file else b""

    # Handle directory input for ball_data
    if isinstance(ball_data, (str, Path)) and Path(ball_data).is_dir():
        ball_data_list = _discover_files_in_directory(ball_data, "*.ball")
    elif isinstance(ball_data, list):
        ball_data_list = ball_data
    else:
        ball_data_list = [ball_data]

    # Handle directory input for player_data
    if isinstance(player_data, (str, Path)) and Path(player_data).is_dir():
        player_data_list = _discover_files_in_directory(player_data, "*.centroids")
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
        filename = _get_filename(ball_file)
        with open_as_file(ball_file) as f:
            ball_bytes_list.append((filename, f.read() if f else b""))

    # Convert player_data to list of (filename, bytes) tuples
    player_bytes_list = []
    for player_file in player_data_list:
        filename = _get_filename(player_file)
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
    )

    return TrackingDataset(
        tracking=tracking_df,
        metadata=metadata_df,
        teams=team_df,
        players=player_df,
        periods=periods_df,
    )


def load_metadata_only(
    meta_data: FileLike,
    player_data: Optional[Union[FileLike, List[FileLike]]] = None,
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal["static_home_away"] = "static_home_away",
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["auto", "heId", "fifaId"] = "auto",
    include_game_id: Union[bool, str] = True,
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
            player_list = _discover_files_in_directory(player_data, "*.centroids")
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
    )
