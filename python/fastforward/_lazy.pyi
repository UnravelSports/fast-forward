"""Type stubs for fastforward._lazy"""

from typing import Dict, List, Union
import polars as pl
from kloppy.io import FileLike


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

    Returns a real pl.LazyFrame with full Polars functionality.
    Data is only loaded when .collect() is called.
    """
    ...


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
) -> pl.LazyFrame:
    """Create a lazy tracking DataFrame for HawkEye's dual-input format.

    Returns a real pl.LazyFrame with full Polars functionality.
    """
    ...
