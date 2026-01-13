"""TrackingDataset class for structured access to tracking data components."""

from typing import Optional, Union
import polars as pl


def extract_players_from_tracking(
    tracking_df: pl.DataFrame,
    periods_df: pl.DataFrame,
    existing_players_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Extract player information from tracking data.

    This function extracts player metadata from tracking data when it's not
    available from the metadata file (e.g., Tracab).

    Parameters
    ----------
    tracking_df : pl.DataFrame
        Tracking DataFrame in long format (must have player_id, team_id columns)
    periods_df : pl.DataFrame
        Periods DataFrame with period_id, start_frame_id
    existing_players_df : pl.DataFrame, optional
        Existing players DataFrame to merge with (for names, jersey_number, etc.)

    Returns
    -------
    pl.DataFrame
        Players DataFrame with standard schema: game_id, team_id, player_id, name,
        first_name, last_name, jersey_number, position, is_starter
    """
    # Columns we want to extract if they exist
    player_cols = ["player_id", "team_id"]
    optional_cols = ["game_id"]

    # Check which optional columns exist in tracking data
    available_optional = [c for c in optional_cols if c in tracking_df.columns]
    select_cols = player_cols + available_optional

    # Filter to exclude ball rows
    players_tracking = tracking_df.filter(pl.col("player_id") != "ball")

    # Get first frame of each period for starter detection
    first_frames = periods_df.select("start_frame_id").to_series().to_list()

    # Get players in first frame of each period (starters)
    starters_df = (
        players_tracking
        .filter(pl.col("frame_id").is_in(first_frames))
        .select(select_cols)
        .unique(subset=["player_id"])
        .with_columns(pl.lit(True).alias("is_starter"))
    )

    # Get all unique players
    all_players_df = (
        players_tracking
        .select(select_cols)
        .unique(subset=["player_id"])
    )

    # Join to get starter flag for all players
    players_df = (
        all_players_df
        .join(
            starters_df.select(["player_id", "is_starter"]),
            on="player_id",
            how="left"
        )
        .with_columns(
            pl.col("is_starter").fill_null(False)
        )
    )

    # If we have existing player data with more info, merge it
    if existing_players_df is not None and existing_players_df.height > 0:
        # Get columns from existing that we don't have
        existing_cols = set(existing_players_df.columns) - set(players_df.columns)
        if existing_cols and "player_id" in existing_players_df.columns:
            merge_cols = ["player_id"] + list(existing_cols)
            players_df = players_df.join(
                existing_players_df.select(merge_cols),
                on="player_id",
                how="left"
            )

    # Ensure standard schema with null placeholders for missing columns
    standard_columns = [
        ("game_id", pl.Utf8),
        ("team_id", pl.Utf8),
        ("player_id", pl.Utf8),
        ("name", pl.Utf8),
        ("first_name", pl.Utf8),
        ("last_name", pl.Utf8),
        ("jersey_number", pl.Int32),
        ("position", pl.Utf8),
        ("is_starter", pl.Boolean),
    ]

    for col_name, col_type in standard_columns:
        if col_name not in players_df.columns:
            players_df = players_df.with_columns(
                pl.lit(None).cast(col_type).alias(col_name)
            )

    # Reorder columns to standard order
    players_df = players_df.select([col for col, _ in standard_columns])

    return players_df


class TrackingDataset:
    """Container for tracking data and associated metadata.

    Attributes
    ----------
    tracking : pl.DataFrame or pl.LazyFrame
        Tracking data. DataFrame if loaded eagerly (lazy=False), LazyFrame if lazy=True.
        For lazy loading, call .collect() to get DataFrame.
        LazyFrame provides full Polars functionality including schema introspection,
        filter pushdown, and all LazyFrame methods (join, group_by, with_columns, etc.).
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

    Lazy loading (default):

    >>> dataset = secondspectrum.load_tracking("tracking.jsonl", "meta.json")
    >>> dataset.tracking  # pl.LazyFrame
    >>> dataset.tracking.schema  # Access schema before collect
    >>> period1 = dataset.tracking.filter(pl.col("period_id") == 1).collect()

    Full LazyFrame functionality:

    >>> result = (
    ...     dataset.tracking
    ...     .filter(pl.col("period_id") == 1)
    ...     .with_columns(pl.col("x") * 100)
    ...     .group_by("player_id")
    ...     .agg(pl.col("x").mean())
    ...     .collect()
    ... )
    """

    def __init__(
        self,
        tracking: Union[pl.DataFrame, pl.LazyFrame],
        metadata: pl.DataFrame,
        teams: pl.DataFrame,
        players: pl.DataFrame,
        periods: pl.DataFrame,
        *,
        _original_tracking: Optional[pl.LazyFrame] = None,
        _provider: Optional[str] = None,
        _cache_key: Optional[str] = None,
    ):
        """Initialize TrackingDataset.

        Parameters
        ----------
        tracking : pl.DataFrame or pl.LazyFrame
            Tracking data. LazyFrame when lazy=True (default), DataFrame when lazy=False.
        metadata : pl.DataFrame
            Metadata (single row)
        teams : pl.DataFrame
            Teams data
        players : pl.DataFrame
            Players data
        periods : pl.DataFrame
            Periods data with period_id, start_frame_id, end_frame_id
        _original_tracking : pl.LazyFrame, optional
            Internal: Original unmodified LazyFrame for cache writes.
        _provider : str, optional
            Internal: Provider name for cache path.
        _cache_key : str, optional
            Internal: Cache key for cache path.
        """
        self._tracking = tracking
        self._metadata = metadata
        self._teams = teams
        self._players = players
        self._periods = periods

        # Internal state for caching
        self._original_tracking = _original_tracking if _original_tracking is not None else tracking
        self._provider = _provider
        self._cache_key = _cache_key
        self._collected_df: Optional[pl.DataFrame] = None

    @property
    def tracking(self) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Get tracking data (DataFrame or LazyFrame).

        Returns the collected DataFrame if collect() was previously called,
        otherwise returns the original LazyFrame.
        """
        if self._collected_df is not None:
            return self._collected_df
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

    def collect(self) -> pl.DataFrame:
        """Collect the tracking LazyFrame into a DataFrame.

        Convenience method that calls self.tracking.collect().
        For lazy datasets, this triggers the actual data loading.

        After calling collect(), subsequent access to dataset.tracking
        will return the collected DataFrame instead of the LazyFrame.

        Returns
        -------
        pl.DataFrame
            The collected tracking data.

        Examples
        --------
        >>> dataset = tracab.load_tracking("raw.dat", "meta.xml")
        >>> df = dataset.collect()  # Same as dataset.tracking.collect()
        >>> dataset.tracking  # Now returns DataFrame, not LazyFrame
        """
        # Return cached result if already collected
        if self._collected_df is not None:
            return self._collected_df

        # If already a DataFrame, cache and return it
        if isinstance(self._tracking, pl.DataFrame):
            self._collected_df = self._tracking
            return self._collected_df

        # Collect LazyFrame, cache, and return
        self._collected_df = self._tracking.collect()
        return self._collected_df

    def collect_with_metadata(self) -> "TrackingDataset":
        """Collect lazy tracking data and populate metadata.

        If tracking is already eager (pl.DataFrame), returns self unchanged.
        If metadata (e.g., players) is incomplete, extracts it from tracking data.

        This method is useful for:
        1. Transitioning from lazy to eager mode with complete metadata
        2. Ensuring players are populated for providers like Tracab where
           player info is embedded in tracking data, not metadata

        Returns
        -------
        TrackingDataset
            New TrackingDataset with eager DataFrame and populated metadata,
            or self if already eager.

        Examples
        --------
        >>> dataset = tracab.load_tracking("raw.dat", "meta.xml", lazy=True)
        >>> dataset.players.height  # May be 0 for Tracab
        0
        >>> eager_dataset = dataset.collect_with_metadata()
        >>> eager_dataset.players.height  # Now populated from tracking data
        22
        """
        if isinstance(self._tracking, pl.DataFrame):
            return self  # Already eager

        # Collect tracking data
        tracking_df = self._tracking.collect()

        # Populate missing players from tracking data
        players_df = self._players
        if players_df.height == 0 and "player_id" in tracking_df.columns:
            players_df = extract_players_from_tracking(
                tracking_df,
                self._periods,
                existing_players_df=self._players,
            )

        return TrackingDataset(
            tracking=tracking_df,
            metadata=self._metadata,
            teams=self._teams,
            players=players_df,
            periods=self._periods,
            _provider=self._provider,
            _cache_key=self._cache_key,
        )

    def write_cache(self) -> None:
        """Write tracking data to cache.

        Caches the original (unmodified) tracking data to the global cache directory.
        The cache directory is configured via kloppy_light.set_cache_dir() or
        the KLOPPY_LIGHT_CACHE_DIR environment variable.

        This method:
        1. Collects the original LazyFrame (not any filtered/modified version)
        2. Extracts player metadata from tracking data if empty
        3. Writes parquet file and metadata sidecar to cache

        Raises
        ------
        RuntimeError
            If cache information is not available (dataset not loaded with
            cache support enabled).

        Examples
        --------
        >>> dataset = tracab.load_tracking("raw.dat", "meta.xml")
        >>> dataset.write_cache()  # Writes to global cache directory
        """
        from kloppy_light._cache import (
            get_cache_path,
            write_cache as cache_write,
        )

        if self._provider is None or self._cache_key is None:
            raise RuntimeError(
                "Cannot write cache: dataset was not loaded with cache support. "
                "Ensure you're using the standard load_tracking() function."
            )

        # Collect original (unmodified) tracking data
        if isinstance(self._original_tracking, pl.LazyFrame):
            tracking_df = self._original_tracking.collect()
        else:
            tracking_df = self._original_tracking

        # Extract players from tracking if not available
        players_df = self._players
        if players_df.height == 0 and "player_id" in tracking_df.columns:
            players_df = extract_players_from_tracking(
                tracking_df,
                self._periods,
                existing_players_df=self._players,
            )

        # Get cache path using global config
        cache_path = get_cache_path(self._cache_key, self._provider)

        # Write to cache
        cache_write(
            tracking_df=tracking_df,
            cache_path=cache_path,
            metadata_df=self._metadata,
            teams_df=self._teams,
            players_df=players_df,
            periods_df=self._periods,
        )

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
