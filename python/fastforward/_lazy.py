"""Lazy loading via Polars register_io_source.

This module provides functions to create real pl.LazyFrame objects for tracking data
using Polars' IO plugin system. The actual data parsing is deferred until .collect().
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import polars as pl
from kloppy.io import FileLike, open_as_file


def _is_local_file(file_like: FileLike) -> bool:
    """Check if a FileLike is a local file (not a remote URL)."""
    if isinstance(file_like, (str, Path)):
        path_str = str(file_like)
        return not path_str.startswith(("s3://", "gs://", "http://", "https://"))
    # For file objects, assume local
    return hasattr(file_like, "name") and not str(getattr(file_like, "name", "")).startswith(
        ("s3://", "gs://", "http://", "https://")
    )


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
    **provider_kwargs,
) -> pl.LazyFrame:
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
    **provider_kwargs
        Provider-specific parameters

    Returns
    -------
    pl.LazyFrame
        Lazy tracking DataFrame
    """
    from fastforward._base import get_provider

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

        # Call Rust to load tracking data with predicate pushdown
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
    include_officials: bool = False,
    parallel: bool = True,
) -> pl.LazyFrame:
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

    Returns
    -------
    pl.LazyFrame
        Lazy tracking DataFrame
    """
    from fastforward._base import get_filename_from_filelike
    from fastforward._fastforward import hawkeye as _hawkeye

    # Only convert to lists eagerly (no file reading)
    ball_list = ball_data if isinstance(ball_data, list) else [ball_data]
    player_list = player_data if isinstance(player_data, list) else [player_data]

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        """Generator that yields HawkEye tracking DataFrame."""
        # Read files LAZILY here, at .collect() time
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

        # Call Rust to load tracking data with predicate pushdown
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
                include_officials=include_officials,
                predicate=predicate,
                parallel=parallel,
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
                    include_officials=include_officials,
                    parallel=parallel,
                )
            else:
                raise

        # Always apply predicate in Python to handle filters Rust couldn't push down
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


def create_lazy_tracking_signality(
    raw_data_feeds: Union[FileLike, List[FileLike]],
    meta_data: FileLike,
    venue_information: FileLike,
    schema: Dict[str, pl.DataType],
    layout: str,
    coordinates: str,
    orientation: str,
    only_alive: bool,
    include_game_id: Union[bool, str],
    include_officials: bool = False,
    parallel: bool = True,
) -> pl.LazyFrame:
    """Create a lazy tracking DataFrame for Signality's multi-file format.

    Signality uses per-period JSON files with metadata and venue info.

    Parameters
    ----------
    raw_data_feeds : FileLike or List[FileLike]
        Raw tracking file(s) (one per period)
    meta_data : FileLike
        Metadata file (teams, players, lineups)
    venue_information : FileLike
        Venue file (pitch dimensions)
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
    include_game_id : bool or str
        Whether to include game_id column
    include_officials : bool
        Whether to include officials
    parallel : bool
        Whether to process files in parallel

    Returns
    -------
    pl.LazyFrame
        Lazy tracking DataFrame
    """
    from fastforward._base import get_filename_from_filelike
    from fastforward._fastforward import signality as _signality

    # Only convert to lists eagerly (no file reading)
    raw_list = raw_data_feeds if isinstance(raw_data_feeds, list) else [raw_data_feeds]

    def signality_source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        """Generator that yields Signality tracking DataFrame."""
        # Read files LAZILY here, at .collect() time
        raw_bytes_list: List[Tuple[str, bytes]] = []
        for raw_file in raw_list:
            with open_as_file(raw_file) as f:
                filename = get_filename_from_filelike(raw_file)
                raw_bytes_list.append((filename, f.read() if f else b""))

        with open_as_file(meta_data) as f:
            meta_bytes = f.read() if f else b""

        with open_as_file(venue_information) as f:
            venue_bytes = f.read() if f else b""

        # Call Rust to load tracking data with predicate pushdown
        try:
            tracking_df, _, _, _, _ = _signality.load_tracking(
                raw_bytes_list,
                meta_bytes,
                venue_bytes,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                include_game_id=include_game_id,
                include_officials=include_officials,
                predicate=predicate,
                parallel=parallel,
            )
        except RuntimeError as e:
            if "deserializing" in str(e) or "BindingsError" in str(e):
                # Fall back to loading without predicate
                tracking_df, _, _, _, _ = _signality.load_tracking(
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
            else:
                raise

        # Always apply predicate in Python to handle filters Rust couldn't push down
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
        io_source=signality_source_generator,
        schema=schema,
    )
