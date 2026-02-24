"""GradientSports (formerly PFF) tracking data loader."""

from typing import Literal, Union

import polars as pl
from kloppy.io import FileLike, open_as_file

from fastforward._fastforward import gradientsports as _gradientsports
from fastforward._dataset import TrackingDataset
from fastforward._errors import with_error_handler
from fastforward._schema import get_tracking_schema


def _create_lazy_tracking_gradientsports(
    raw_data: FileLike,
    meta_data: FileLike,
    roster_data: FileLike,
    schema: dict,
    layout: str,
    coordinates: str,
    orientation: str,
    only_alive: bool,
    include_incomplete_frames: bool,
    include_game_id,
) -> pl.LazyFrame:
    """Create a lazy frame for GradientSports tracking data."""

    def load_fn(with_columns=None, n_rows=None, predicate=None):
        # Load all files as bytes
        with open_as_file(raw_data) as raw_file:
            raw_bytes = raw_file.read() if raw_file else b""
        with open_as_file(meta_data) as meta_file:
            meta_bytes = meta_file.read() if meta_file else b""
        with open_as_file(roster_data) as roster_file:
            roster_bytes = roster_file.read() if roster_file else b""

        tracking_df, _, _, _, _ = _gradientsports.load_tracking(
            raw_bytes,
            meta_bytes,
            roster_bytes,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            include_incomplete_frames=include_incomplete_frames,
            include_game_id=include_game_id,
            predicate=predicate,
        )

        # Apply column selection if needed
        if with_columns is not None:
            tracking_df = tracking_df.select(with_columns)

        # Apply row limit if needed
        if n_rows is not None:
            tracking_df = tracking_df.head(n_rows)

        return tracking_df

    return pl.LazyFrame(
        schema=schema,
    ).map_batches(lambda _: load_fn(), schema=schema)


@with_error_handler
def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
    roster_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "gradientsports",
        "cdf",
        "kloppy",
        "opta",
        "secondspectrum",
        "skillcorner",
        "sportec:event",
        "sportec:tracking",
        "sportvu",
        "tracab",
        "pff",
        "hawkeye",
    ] = "gradientsports",
    orientation: Literal[
        "static_home_away",
        "attack_left",
        "attack_right",
        "away_home",
        "home_away",
        "static_away_home",
    ] = "static_home_away",
    only_alive: bool = True,
    include_incomplete_frames: bool = False,
    include_game_id: Union[bool, str] = True,
    *,
    lazy: bool = False,
) -> TrackingDataset:
    """
    Load GradientSports (PFF) tracking data.

    Parameters
    ----------
    raw_data : FileLike
        Path to JSONL tracking file
    meta_data : FileLike
        Path to JSON metadata file
    roster_data : FileLike
        Path to JSON roster file
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
    coordinates : str, default "gradientsports"
        Coordinate system (gradientsports uses CDF format natively)
    orientation : str, default "static_home_away"
        Coordinate orientation
    only_alive : bool, default True
        If True, only include frames where ball is in play
    include_incomplete_frames : bool, default False
        If True, include frames with null ball coordinates or null player arrays.
        If False (default), only include frames with complete data.
    include_game_id : Union[bool, str], default True
        If True, add game_id column from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    Returns
    -------
    TrackingDataset
        Object with .tracking, .metadata, .teams, .players, .periods properties.
    """
    if lazy:
        raise NotImplementedError("lazy loading is not yet supported in fast-forward")
        # Load metadata only
        with open_as_file(meta_data) as meta_file:
            meta_bytes = meta_file.read() if meta_file else b""
        with open_as_file(roster_data) as roster_file:
            roster_bytes = roster_file.read() if roster_file else b""

        metadata_df, team_df, player_df, periods_df = _gradientsports.load_metadata_only(
            meta_bytes,
            roster_bytes,
            coordinates=coordinates,
            orientation=orientation,
            include_game_id=include_game_id,
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
            with open_as_file(raw_data) as raw_file:
                raw_bytes = raw_file.read() if raw_file else b""
            with open_as_file(meta_data) as meta_file2:
                meta_bytes2 = meta_file2.read() if meta_file2 else b""
            with open_as_file(roster_data) as roster_file2:
                roster_bytes2 = roster_file2.read() if roster_file2 else b""

            tracking_df, _, _, _, _ = _gradientsports.load_tracking(
                raw_bytes,
                meta_bytes2,
                roster_bytes2,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                include_incomplete_frames=include_incomplete_frames,
                include_game_id=include_game_id,
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
            _provider="gradientsports",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )
    else:
        # Eager loading - load all files as bytes
        with open_as_file(raw_data) as raw_file:
            raw_bytes = raw_file.read() if raw_file else b""
        with open_as_file(meta_data) as meta_file:
            meta_bytes = meta_file.read() if meta_file else b""
        with open_as_file(roster_data) as roster_file:
            roster_bytes = roster_file.read() if roster_file else b""

        tracking_df, metadata_df, team_df, player_df, periods_df = (
            _gradientsports.load_tracking(
                raw_bytes,
                meta_bytes,
                roster_bytes,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                include_incomplete_frames=include_incomplete_frames,
                include_game_id=include_game_id,
            )
        )

        return TrackingDataset(
            tracking=tracking_df,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
            _engine="polars",
            _provider="gradientsports",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )
