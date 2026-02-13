"""Respovision provider wrapper.

Respovision tracking data consists of a single JSONL file with all metadata embedded.
There is no separate metadata file - team names, player info, and coordinates
are all included in each frame.

Note: Lazy loading is NOT supported because metadata must be extracted from
the tracking file, requiring a full parse.
"""

from typing import TYPE_CHECKING, Literal, Optional, Union

from kloppy.io import FileLike, open_as_file

from fastforward._base import get_filename_from_filelike
from fastforward._dataset import TrackingDataset

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def load_tracking(
    raw_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "kloppy",
        "opta",
        "pff",
        "respovision",
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
    exclude_missing_ball_frames: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    include_game_id: Union[bool, str] = True,
    include_joint_angles: bool = True,
    include_officials: bool = False,
    *,
    lazy: bool = False,
    engine: Literal["polars", "pyspark"] = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """
    Load Respovision tracking data.

    Respovision data comes in a single JSONL file containing all tracking frames
    with embedded metadata. Team names are extracted from the filename pattern
    YYYYMMDD-HomeTeam-AwayTeam-*.jsonl.

    Parameters
    ----------
    raw_data : FileLike
        Path to JSONL tracking file, or bytes, or file-like object.
        Filename pattern: YYYYMMDD-HomeTeam-AwayTeam-*.jsonl
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths.
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
        Note: Wide layout does not include joint angles.
    coordinates : str, default "cdf"
        Coordinate system. Options:
        - "cdf": Common Data Format (origin at center, meters)
        - "respovision": Native coordinates (origin at bottom-left corner, meters)
        - Other provider coordinate systems
    orientation : str, default "static_home_away"
        Coordinate orientation:
        - "static_home_away": Home attacks right (+x) entire match
        - Other orientations available
    only_alive : bool, default True
        If True, only include frames where ball_possession is not null.
    exclude_missing_ball_frames : bool, default True
        If True, exclude frames where ball coordinates are missing (null).
        Respovision data may have frames where ball tracking failed.
    pitch_length : float, default 105.0
        Pitch length in meters. Used for coordinate transformation.
    pitch_width : float, default 68.0
        Pitch width in meters. Used for coordinate transformation.
    include_game_id : bool or str, default True
        If True, add game_id column (auto-generated from filename).
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_joint_angles : bool, default True
        If True, include head_angle, shoulders_angle, hips_angle columns.
        Only applies to long and long_ball layouts.
    include_officials : bool, default False
        If True, include referees in tracking data with team_id="officials".
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
    - Native coordinate system (respovision): origin at bottom-left corner, meters
      X in [0, pitch_length], Y in [0, pitch_width]
    - Home/away team designation is extracted from filename
    - Player IDs are formatted as {team_name_lower}_{jersey_number}
    - Team IDs are lowercase team names with spaces replaced by underscores
    - Game ID default format: YYYYMMDD-{home_prefix}-{away_prefix}
    - Frame rate is typically 25 Hz
    - Ball state: alive if ball_possession is not null, dead otherwise
    - Joint angles may contain null values (especially for goalkeepers)

    Examples
    --------
    Load from file path:

    >>> from fastforward import respovision
    >>> dataset = respovision.load_tracking(
    ...     "20240714-Argentina-Colombia-2d_tracking-tactical.jsonl",
    ...     pitch_length=105.0,
    ...     pitch_width=68.0,
    ... )
    >>> tracking_df = dataset.tracking

    Load without joint angles:

    >>> dataset = respovision.load_tracking(
    ...     "tracking.jsonl",
    ...     include_joint_angles=False,
    ... )
    """
    from fastforward._engine import (
        validate_engine,
        polars_to_spark,
        get_spark_session,
    )
    from fastforward._fastforward import respovision as _respovision

    # Validate engine parameter
    engine = validate_engine(engine)

    # Respovision does NOT support lazy loading
    if lazy:
        raise ValueError(
            "lazy=True is not supported for Respovision. "
            "Metadata is embedded in the tracking file and cannot be "
            "populated without parsing the entire file."
        )

    # For PySpark, force eager loading (will convert after)
    if engine == "pyspark":
        lazy = False

    # Extract filename for metadata extraction
    filename = get_filename_from_filelike(raw_data)

    # Read raw data
    with open_as_file(raw_data) as raw_file:
        raw_bytes = raw_file.read() if raw_file else b""

    # Load tracking data
    tracking_df, metadata_df, team_df, player_df, periods_df = (
        _respovision.load_tracking(
            raw_bytes,
            filename=filename,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            exclude_missing_ball_frames=exclude_missing_ball_frames,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            include_game_id=include_game_id,
            include_joint_angles=include_joint_angles,
            include_officials=include_officials,
        )
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
            _provider="respovision",
            _cache_key=None,
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
        _provider="respovision",
        _cache_key=None,
        _coordinate_system=coordinates,
        _orientation=orientation,
    )
