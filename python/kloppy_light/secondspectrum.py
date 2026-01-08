"""SecondSpectrum provider wrapper with lazy loading support."""

from typing import Literal, Tuple, Union, overload
import polars as pl

from kloppy_light._kloppy_light import secondspectrum as _secondspectrum
from kloppy_light._lazy import LazyTrackingLoader


@overload
def load_tracking(
    raw_data: str,
    meta_data: str,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal[
        "static_home_away",
        "static_away_home",
        "home_away",
        "away_home",
        "attack_right",
        "attack_left",
    ] = "static_home_away",
    only_alive: bool = False,
    *,
    lazy: Literal[False] = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]: ...


@overload
def load_tracking(
    raw_data: str,
    meta_data: str,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal[
        "static_home_away",
        "static_away_home",
        "home_away",
        "away_home",
        "attack_right",
        "attack_left",
    ] = "static_home_away",
    only_alive: bool = False,
    *,
    lazy: Literal[True],
) -> Tuple[LazyTrackingLoader, pl.DataFrame, pl.DataFrame, pl.DataFrame]: ...


def load_tracking(
    raw_data: str,
    meta_data: str,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal[
        "static_home_away",
        "static_away_home",
        "home_away",
        "away_home",
        "attack_right",
        "attack_left",
    ] = "static_home_away",
    only_alive: bool = False,
    *,
    lazy: bool = False,
) -> Union[
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
    Tuple[LazyTrackingLoader, pl.DataFrame, pl.DataFrame, pl.DataFrame],
]:
    """
    Load SecondSpectrum tracking data.

    Parameters
    ----------
    raw_data : str
        Path to JSONL tracking file
    meta_data : str
        Path to JSON metadata file
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
    coordinates : {"cdf"}, default "cdf"
        Coordinate system:
        - "cdf": Common Data Format (origin at center)
    orientation : str, default "static_home_away"
        Coordinate orientation:
        - "static_home_away": Home attacks right (+x) entire match
        - "static_away_home": Away attacks right (+x) entire match
        - "home_away": Home attacks right 1st half, left 2nd half
        - "away_home": Away attacks right 1st half, left 2nd half
        - "attack_right": Attacking team always attacks right
        - "attack_left": Attacking team always attacks left
    only_alive : bool, default False
        If True, only include frames where ball is in play (ball_state == "alive")
    lazy : bool, default False
        If True, return a LazyTrackingLoader that defers parsing until .collect()

    Returns
    -------
    If lazy=False:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (tracking_df, metadata_df, team_df, player_df)
    If lazy=True:
        Tuple[LazyTrackingLoader, pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (tracking_lazy, metadata_df, team_df, player_df)
            The tracking_lazy object supports .filter(), .select(), and .collect()
    """
    if lazy:
        # Get metadata, team, and player DataFrames eagerly
        # but defer tracking data parsing
        _, metadata_df, team_df, player_df = _secondspectrum.load_tracking(
            raw_data,
            meta_data,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
        )

        lazy_loader = LazyTrackingLoader(
            provider="secondspectrum",
            raw_data=raw_data,
            meta_data=meta_data,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
        )

        return lazy_loader, metadata_df, team_df, player_df
    else:
        return _secondspectrum.load_tracking(
            raw_data,
            meta_data,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
        )
