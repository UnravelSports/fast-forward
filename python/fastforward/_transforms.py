"""Coordinate, orientation, and dimension transformation functions.

These functions wrap the Rust implementations for post-load transformation
of tracking data.
"""

import polars as pl
from fastforward._fastforward import transforms as _rust_transforms


def transform_coordinates(
    df: pl.DataFrame,
    from_system: str,
    to_system: str,
    pitch_length: float,
    pitch_width: float,
) -> pl.DataFrame:
    """Transform DataFrame coordinates between coordinate systems.

    Uses CDF as intermediate format: source -> CDF -> target.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with x, y columns (and optionally z)
    from_system : str
        Source coordinate system (e.g., "cdf", "tracab", "opta")
    to_system : str
        Target coordinate system
    pitch_length : float
        Pitch length in meters
    pitch_width : float
        Pitch width in meters

    Returns
    -------
    pl.DataFrame
        DataFrame with transformed x, y, z columns
    """
    return _rust_transforms.transform_coordinates(
        df, from_system, to_system, pitch_length, pitch_width
    )


def transform_dimensions(
    df: pl.DataFrame,
    from_length: float,
    from_width: float,
    to_length: float,
    to_width: float,
) -> pl.DataFrame:
    """Transform DataFrame to different pitch dimensions using zone-based scaling.

    Uses IFAB standard zone boundaries to preserve pitch feature proportions
    (penalty area, six-yard box, center circle, etc.).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with x, y columns (must be in CDF format: center origin, meters)
    from_length : float
        Source pitch length in meters
    from_width : float
        Source pitch width in meters
    to_length : float
        Target pitch length in meters
    to_width : float
        Target pitch width in meters

    Returns
    -------
    pl.DataFrame
        DataFrame with zone-scaled x, y coordinates
    """
    return _rust_transforms.transform_dimensions(
        df, from_length, from_width, to_length, to_width
    )


def transform_orientation(
    df: pl.DataFrame,
    flip: bool,
) -> pl.DataFrame:
    """Transform DataFrame orientation by flipping coordinates.

    Orientation flipping negates x and y coordinates around the center (0, 0).
    This is used to ensure consistent attacking direction.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with x, y columns (must be in CDF format: center origin)
    flip : bool
        If True, flip the coordinates (negate x and y)

    Returns
    -------
    pl.DataFrame
        DataFrame with flipped x, y coordinates (if flip=True)
    """
    return _rust_transforms.transform_orientation(df, flip)
