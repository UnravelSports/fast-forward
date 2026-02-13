"""Schema generation for tracking DataFrames.

This module provides schema definitions for use with pl.io.plugins.register_io_source().
The schema is required upfront to create a LazyFrame before data is loaded.
"""

from typing import Dict, Literal
import polars as pl


# Base columns present in all layouts
_BASE_SCHEMA: Dict[str, pl.DataType] = {
    "frame_id": pl.UInt32,
    "period_id": pl.Int32,
    "timestamp": pl.Duration("ms"),
    "ball_state": pl.String,
    "ball_owning_team_id": pl.String,
}


def get_tracking_schema(
    layout: Literal["long", "long_ball", "wide"],
    players_df: pl.DataFrame = None,
    include_game_id: bool = True,
) -> Dict[str, pl.DataType]:
    """Generate tracking DataFrame schema based on layout and players.

    Parameters
    ----------
    layout : {"long", "long_ball", "wide"}
        DataFrame layout
    players_df : pl.DataFrame, optional
        Players DataFrame with player_id column. Required for "wide" layout.
    include_game_id : bool, default True
        Whether to include game_id column

    Returns
    -------
    dict
        Schema dict mapping column names to Polars data types

    Examples
    --------
    >>> schema = get_tracking_schema("long", include_game_id=True)
    >>> schema
    {'game_id': String, 'frame_id': UInt32, ...}
    """
    schema = {}

    # Add game_id first if requested
    if include_game_id:
        schema["game_id"] = pl.String

    # Add base columns
    schema.update(_BASE_SCHEMA)

    if layout == "long":
        # Long format: team_id, player_id, x, y, z
        # Ball is included as row with team_id="ball", player_id="ball"
        schema.update({
            "team_id": pl.String,
            "player_id": pl.String,
            "x": pl.Float32,
            "y": pl.Float32,
            "z": pl.Float32,
        })
    elif layout == "long_ball":
        # Long ball format: ball separate from players
        # Ball columns at frame level, player rows for players only
        schema.update({
            "ball_x": pl.Float32,
            "ball_y": pl.Float32,
            "ball_z": pl.Float32,
            "team_id": pl.String,
            "player_id": pl.String,
            "x": pl.Float32,
            "y": pl.Float32,
            "z": pl.Float32,
        })
    elif layout == "wide":
        # Wide format: one row per frame, player columns
        # Ball columns
        schema.update({
            "ball_x": pl.Float32,
            "ball_y": pl.Float32,
            "ball_z": pl.Float32,
        })

        # Player columns: {player_id}_x, {player_id}_y, {player_id}_z
        if players_df is not None and "player_id" in players_df.columns:
            player_ids = sorted(players_df["player_id"].unique().to_list())
            for player_id in player_ids:
                schema[f"{player_id}_x"] = pl.Float32
                schema[f"{player_id}_y"] = pl.Float32
                schema[f"{player_id}_z"] = pl.Float32
        # If no players_df, schema will be incomplete for wide layout
        # This is handled by register_io_source's validate_schema option
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return schema


def get_long_schema(include_game_id: bool = True) -> Dict[str, pl.DataType]:
    """Get schema for long format (most common).

    Convenience function for the default long layout.

    Parameters
    ----------
    include_game_id : bool, default True
        Whether to include game_id column

    Returns
    -------
    dict
        Schema dict for long format
    """
    return get_tracking_schema("long", include_game_id=include_game_id)
