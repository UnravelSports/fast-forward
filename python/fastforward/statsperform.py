"""StatsPerform (Opta) tracking data loader.

This module provides functions for loading StatsPerform tracking data,
supporting both MA25 tracking files and MA1 metadata (JSON or XML format).

Example
-------
    from fastforward import statsperform

    # Load tracking data with MA1 JSON metadata
    dataset = statsperform.load_tracking(
        ma25_data="tracking.txt",
        ma1_data="metadata.json",
        pitch_length=105.0,
        pitch_width=68.0,
    )

    # Access the data
    print(dataset.tracking)
    print(dataset.metadata)
    print(dataset.players)
"""

from typing import Literal, Optional, Union

import polars as pl
from kloppy.io import FileLike, open_as_file

from fastforward._fastforward import statsperform as _statsperform
from fastforward._dataset import TrackingDataset
from fastforward._errors import with_error_handler
from fastforward._schema import get_tracking_schema


@with_error_handler
def load_tracking(
    ma25_data: FileLike,
    ma1_data: FileLike,
    pitch_length: Optional[float] = None,
    pitch_width: Optional[float] = None,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "statsperform",
        "sportvu",
        "kloppy",
        "opta",
        "secondspectrum",
        "skillcorner",
        "sportec:event",
        "sportec:tracking",
        "tracab",
        "pff",
        "gradientsports",
        "hawkeye",
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
) -> TrackingDataset:
    """
    Load StatsPerform tracking data.

    Parameters
    ----------
    ma25_data : FileLike
        Path to MA25 tracking data file (text format).
    ma1_data : FileLike
        Path to MA1 metadata file (JSON or XML format, auto-detected).
    pitch_length : float, optional
        Length of the pitch in meters. StatsPerform data does not include
        pitch dimensions, so this must be provided. Default: 105.0m.
    pitch_width : float, optional
        Width of the pitch in meters. StatsPerform data does not include
        pitch dimensions, so this must be provided. Default: 68.0m.
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
    coordinates : str, default "cdf"
        Coordinate system for output. Options:
        - "cdf": Center origin, meters (default)
        - "statsperform" / "sportvu": Native top-left origin, y-down, meters
        - Other provider coordinate systems
    orientation : str, default "static_home_away"
        Coordinate orientation
    only_alive : bool, default True
        If True, only include frames where ball is in play
    include_game_id : Union[bool, str], default True
        If True, add game_id column from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_officials : bool, default False
        If True, include match officials (referees) in the players DataFrame
        with team_id="officials" and appropriate position codes (REF, AREF, 4TH).
    Returns
    -------
    TrackingDataset
        Object with .tracking, .metadata, .teams, .players, .periods properties.

    Notes
    -----
    StatsPerform uses the SportVU coordinate system:
    - Origin at top-left corner of the pitch
    - X increases left to right (0 to ~105m)
    - Y increases top to bottom (0 to ~68m) - inverted from standard
    - Units are meters
    - Frame rate is typically 10 Hz (100ms between frames)

    The MA1 metadata format is auto-detected (JSON or XML) based on content.
    """
    if lazy:
        raise NotImplementedError("lazy loading is not yet supported in fast-forward")

    # Wide format doesn't support lazy loading - column names are game-specific
    if lazy and layout == "wide":
        raise ValueError(
            "lazy=True is not supported for layout='wide'. "
            "Wide format has game-specific column names (player IDs), "
            "making lazy frame operations like concatenation incompatible."
        )

    if lazy:
        # Load metadata only
        with open_as_file(ma1_data) as ma1_file:
            ma1_bytes = ma1_file.read() if ma1_file else b""

        metadata_df, team_df, player_df, periods_df = _statsperform.load_metadata_only(
            ma1_bytes,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            coordinates=coordinates,
            orientation=orientation,
            include_game_id=include_game_id,
            include_officials=include_officials,
        )

        # Generate schema
        schema = get_tracking_schema(
            layout=layout,
            players_df=player_df,
            include_game_id=bool(include_game_id),
        )

        # Create lazy frame using register_io_source
        def source_fn(with_columns=None, n_rows=None, predicate=None):
            # Load all files as bytes
            with open_as_file(ma25_data) as ma25_file:
                ma25_bytes = ma25_file.read() if ma25_file else b""
            with open_as_file(ma1_data) as ma1_file2:
                ma1_bytes2 = ma1_file2.read() if ma1_file2 else b""

            tracking_df, _, _, _, _ = _statsperform.load_tracking(
                ma25_bytes,
                ma1_bytes2,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                include_game_id=include_game_id,
                include_officials=include_officials,
                predicate=predicate,
            )

            # Apply column selection if needed
            if with_columns is not None:
                tracking_df = tracking_df.select(with_columns)

            # Apply row limit if needed
            if n_rows is not None:
                tracking_df = tracking_df.head(n_rows)

            return tracking_df

        lazy_frame = pl.LazyFrame(
            schema=schema,
        ).map_batches(
            lambda df: source_fn(),
            schema=schema,
        )

        return TrackingDataset(
            tracking=lazy_frame,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
            _engine="polars",
            _provider="statsperform",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )
    else:
        # Eager loading - load all files as bytes
        with open_as_file(ma25_data) as ma25_file:
            ma25_bytes = ma25_file.read() if ma25_file else b""
        with open_as_file(ma1_data) as ma1_file:
            ma1_bytes = ma1_file.read() if ma1_file else b""

        tracking_df, metadata_df, team_df, player_df, periods_df = (
            _statsperform.load_tracking(
                ma25_bytes,
                ma1_bytes,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                include_game_id=include_game_id,
                include_officials=include_officials,
            )
        )

        return TrackingDataset(
            tracking=tracking_df,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
            _engine="polars",
            _provider="statsperform",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )


@with_error_handler
def load_metadata_only(
    ma1_data: FileLike,
    pitch_length: Optional[float] = None,
    pitch_width: Optional[float] = None,
    coordinates: str = "cdf",
    orientation: str = "static_home_away",
    include_game_id: Union[bool, str] = True,
    include_officials: bool = False,
) -> TrackingDataset:
    """
    Load only StatsPerform metadata without tracking data.

    Useful for quick metadata inspection without parsing large tracking files.

    Parameters
    ----------
    ma1_data : FileLike
        Path to MA1 metadata file (JSON or XML format, auto-detected).
    pitch_length : float, optional
        Length of the pitch in meters. Default: 105.0m.
    pitch_width : float, optional
        Width of the pitch in meters. Default: 68.0m.
    coordinates : str, default "cdf"
        Coordinate system for metadata output.
    orientation : str, default "static_home_away"
        Orientation for metadata output.
    include_game_id : Union[bool, str], default True
        If True, include game_id column from metadata.
    include_officials : bool, default False
        If True, include match officials (referees) in the players DataFrame.

    Returns
    -------
    TrackingDataset
        Dataset with metadata, teams, players, periods (tracking will be empty).
    """
    with open_as_file(ma1_data) as ma1_file:
        ma1_bytes = ma1_file.read() if ma1_file else b""

    metadata_df, team_df, player_df, periods_df = _statsperform.load_metadata_only(
        ma1_bytes,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        coordinates=coordinates,
        orientation=orientation,
        include_game_id=include_game_id,
        include_officials=include_officials,
    )

    # Create empty tracking DataFrame with correct schema
    tracking_df = pl.DataFrame()

    return TrackingDataset(
        tracking=tracking_df,
        metadata=metadata_df,
        teams=team_df,
        players=player_df,
        periods=periods_df,
        _engine="polars",
        _provider="statsperform",
        _cache_key=None,
        _coordinate_system=coordinates,
        _orientation=orientation,
    )
