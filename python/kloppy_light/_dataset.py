"""TrackingDataset class for structured access to tracking data components."""

from typing import Union
import polars as pl
from kloppy_light._lazy import LazyTrackingLoader


class TrackingDataset:
    """Container for tracking data and associated metadata.

    Attributes
    ----------
    tracking : pl.DataFrame or LazyTrackingLoader
        Tracking data. DataFrame if loaded eagerly, LazyTrackingLoader if lazy=True.
        For lazy loading, call .collect() to get DataFrame.
    metadata : pl.DataFrame
        Single-row DataFrame with match-level metadata
    teams : pl.DataFrame
        Team information (2 rows: home and away)
    players : pl.DataFrame
        Player information with team associations
    periods : pl.DataFrame
        Period information with period_id, start_frame_id, end_frame_id

    Examples
    --------
    Eager loading:

    >>> from kloppy_light import secondspectrum
    >>> dataset = secondspectrum.load_tracking("tracking.jsonl", "meta.json", lazy=False)
    >>> dataset.tracking  # pl.DataFrame
    >>> dataset.metadata  # pl.DataFrame (1 row)
    >>> dataset.periods   # pl.DataFrame (2+ rows)

    Lazy loading:

    >>> dataset = secondspectrum.load_tracking("tracking.jsonl", "meta.json")
    >>> dataset.tracking  # LazyTrackingLoader
    >>> period1 = dataset.tracking.filter(pl.col("period_id") == 1).collect()
    """

    def __init__(
        self,
        tracking: Union[pl.DataFrame, LazyTrackingLoader],
        metadata: pl.DataFrame,
        teams: pl.DataFrame,
        players: pl.DataFrame,
        periods: pl.DataFrame,
    ):
        """Initialize TrackingDataset.

        Parameters
        ----------
        tracking : pl.DataFrame or LazyTrackingLoader
            Tracking data
        metadata : pl.DataFrame
            Metadata (single row)
        teams : pl.DataFrame
            Teams data
        players : pl.DataFrame
            Players data
        periods : pl.DataFrame
            Periods data with period_id, start_frame_id, end_frame_id
        """
        self._tracking = tracking
        self._metadata = metadata
        self._teams = teams
        self._players = players
        self._periods = periods

    @property
    def tracking(self) -> Union[pl.DataFrame, LazyTrackingLoader]:
        """Get tracking data (DataFrame or LazyTrackingLoader)."""
        return self._tracking

    @property
    def metadata(self) -> pl.DataFrame:
        """Get metadata DataFrame (single row)."""
        return self._metadata

    @property
    def teams(self) -> pl.DataFrame:
        """Get teams DataFrame."""
        return self._teams

    @property
    def players(self) -> pl.DataFrame:
        """Get players DataFrame."""
        return self._players

    @property
    def periods(self) -> pl.DataFrame:
        """Get periods DataFrame with period_id, start_frame_id, end_frame_id."""
        return self._periods

    def __repr__(self) -> str:
        """String representation."""
        tracking_type = type(self._tracking).__name__
        game_id = self._metadata["game_id"][0] if "game_id" in self._metadata.columns else "unknown"
        n_periods = len(self._periods)
        n_players = len(self._players)

        return (
            f"TrackingDataset(\n"
            f"  game_id={game_id!r},\n"
            f"  tracking={tracking_type},\n"
            f"  periods={n_periods},\n"
            f"  players={n_players}\n"
            f")"
        )
