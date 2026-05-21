"""OptaVision provider wrapper.

OptaVision is StatsPerform's FIFA Data Transfer Format EPTS export — XML
metadata plus a colon/semicolon-separated text tracking file at the per-frame
level. See `rust/src/providers/optavision.rs` for the full format description.

Note on ball state: OptaVision exports only contain in-play frames. The
`only_alive` parameter is accepted for API parity with the other providers,
but has no effect for this loader — every parsed frame is treated as alive.
"""

from typing import TYPE_CHECKING, Literal, Optional, Union

from kloppy.io import FileLike

from fastforward._base import load_tracking_impl as _load_tracking_impl
from fastforward._dataset import TrackingDataset

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def load_tracking(
    raw_data: FileLike,
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
    include_game_id: Union[bool, str] = True,
    include_ball_owning_player: bool = True,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    engine: Literal["polars", "pyspark"] = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """Load OptaVision (StatsPerform FIFA EPTS) tracking data.

    Parameters
    ----------
    raw_data : FileLike
        Path to the tracking text file (e.g. ``*-trackingdata.txt``), bytes, or
        a file-like object.
    meta_data : FileLike
        Path to the FIFA EPTS metadata XML file (e.g. ``*-metadata.xml``).
    layout : {"long", "long_ball", "wide"}, default "long"
        Output DataFrame layout (see other providers for details).
    coordinates : str, default "cdf"
        Target coordinate system. OptaVision is natively centered metres, so
        the default ``"cdf"`` is the identity transform.
    orientation : str, default "static_home_away"
        Target orientation. The home team is determined by the order in which
        teams appear in the metadata ``<Teams>`` block — first listed is home.
    only_alive : bool, default True
        Accepted for API parity but has no effect for OptaVision. The provider's
        export already excludes out-of-play frames, so every parsed frame is
        treated as alive regardless of this flag.
    include_game_id : bool or str, default True
        Whether to include a ``game_id`` column. ``True`` uses the metadata's
        ``match_uuid``; ``False`` omits the column; a string overrides the
        value.
    include_ball_owning_player : bool, default True
        If True, attach a ``ball_owning_player_id`` column to the tracking
        DataFrame. OptaVision is the only provider that exposes this today;
        the value is the player UUID (matching ``player_id`` in ``ds.players``)
        of the player currently in possession of the ball, or null when the
        export didn't record possession on that frame.
    lazy, from_cache, engine, spark_session
        Standard cross-provider parameters; see other providers.

    Returns
    -------
    TrackingDataset
        With ``.tracking``, ``.metadata``, ``.teams``, ``.players``, ``.periods``.
    """
    return _load_tracking_impl(
        provider_name="optavision",
        raw_data=raw_data,
        meta_data=meta_data,
        layout=layout,
        coordinates=coordinates,
        orientation=orientation,
        only_alive=only_alive,
        include_game_id=include_game_id,
        lazy=lazy,
        from_cache=from_cache,
        engine=engine,
        spark_session=spark_session,
        include_ball_owning_player=include_ball_owning_player,
    )
