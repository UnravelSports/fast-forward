"""TrackingDataset class for structured access to tracking data components."""

from typing import TYPE_CHECKING, Optional, Tuple, Union
import polars as pl

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


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

    Supports multiple DataFrame backends:
    - Polars (default): pl.DataFrame
    - PySpark: pyspark.sql.DataFrame

    Attributes
    ----------
    tracking : pl.DataFrame or pyspark.sql.DataFrame
        Tracking data. DataFrame type depends on engine parameter.
    metadata : pl.DataFrame or pyspark.sql.DataFrame
        Single-row DataFrame with match-level metadata.
    teams : pl.DataFrame or pyspark.sql.DataFrame
        Team information (2 rows: home and away).
    players : pl.DataFrame or pyspark.sql.DataFrame
        Player information with team associations.
    periods : pl.DataFrame or pyspark.sql.DataFrame
        Period information with period_id, start_frame_id, end_frame_id.
    engine : str
        The DataFrame engine being used ('polars' or 'pyspark').

    Examples
    --------
    >>> from fastforward import secondspectrum
    >>> dataset = secondspectrum.load_tracking("tracking.jsonl", "meta.json")
    >>> dataset.tracking  # pl.DataFrame
    >>> dataset.metadata  # pl.DataFrame (1 row)
    >>> dataset.periods   # pl.DataFrame (2+ rows)
    """

    def __init__(
        self,
        tracking: Union[pl.DataFrame, "SparkDataFrame"],
        metadata: Union[pl.DataFrame, "SparkDataFrame"],
        teams: Union[pl.DataFrame, "SparkDataFrame"],
        players: Union[pl.DataFrame, "SparkDataFrame"],
        periods: Union[pl.DataFrame, "SparkDataFrame"],
        *,
        _engine: str = "polars",
        _original_tracking: Optional[pl.LazyFrame] = None,
        _provider: Optional[str] = None,
        _cache_key: Optional[str] = None,
        _coordinate_system: str = "cdf",
        _orientation: str = "static_home_away",
        _pitch_length: Optional[float] = None,
        _pitch_width: Optional[float] = None,
    ):
        """Initialize TrackingDataset.

        Parameters
        ----------
        tracking : pl.DataFrame or pyspark.sql.DataFrame
            Tracking data. Type depends on engine parameter.
        metadata : pl.DataFrame or pyspark.sql.DataFrame
            Metadata (single row)
        teams : pl.DataFrame or pyspark.sql.DataFrame
            Teams data
        players : pl.DataFrame or pyspark.sql.DataFrame
            Players data
        periods : pl.DataFrame or pyspark.sql.DataFrame
            Periods data with period_id, start_frame_id, end_frame_id
        _engine : str, default "polars"
            Internal: DataFrame engine ('polars' or 'pyspark').
        _original_tracking : pl.LazyFrame, optional
            Internal: Original unmodified LazyFrame for cache writes.
        _provider : str, optional
            Internal: Provider name for cache path.
        _cache_key : str, optional
            Internal: Cache key for cache path.
        _coordinate_system : str, default "cdf"
            Internal: Current coordinate system.
        _orientation : str, default "static_home_away"
            Internal: Current orientation.
        _pitch_length : float, optional
            Internal: Current pitch length in meters.
        _pitch_width : float, optional
            Internal: Current pitch width in meters.
        """
        self._tracking = tracking
        self._metadata = metadata
        self._teams = teams
        self._players = players
        self._periods = periods
        self._engine = _engine

        # Internal state for caching
        self._original_tracking = _original_tracking if _original_tracking is not None else tracking
        self._provider = _provider
        self._cache_key = _cache_key
        self._collected_df: Optional[pl.DataFrame] = None

        # Transformation state
        self._coordinate_system = _coordinate_system
        self._orientation = _orientation

        # Get pitch dimensions from metadata if not provided
        if _pitch_length is not None and _pitch_width is not None:
            self._pitch_length = _pitch_length
            self._pitch_width = _pitch_width
        elif self._engine == "polars" and "pitch_length" in metadata.columns:
            self._pitch_length = float(metadata["pitch_length"][0])
            self._pitch_width = float(metadata["pitch_width"][0])
        else:
            # Default IFAB dimensions
            self._pitch_length = 105.0
            self._pitch_width = 68.0

    @property
    def engine(self) -> str:
        """Get the DataFrame engine ('polars' or 'pyspark')."""
        return self._engine

    @property
    def tracking(self) -> Union[pl.DataFrame, "SparkDataFrame"]:
        """Get tracking data as a DataFrame."""
        if self._collected_df is not None:
            return self._collected_df
        return self._tracking

    @property
    def metadata(self) -> Union[pl.DataFrame, "SparkDataFrame"]:
        """Get metadata DataFrame (single row)."""
        return self._metadata

    @property
    def teams(self) -> Union[pl.DataFrame, "SparkDataFrame"]:
        """Get teams DataFrame."""
        return self._teams

    @property
    def players(self) -> Union[pl.DataFrame, "SparkDataFrame"]:
        """Get players DataFrame."""
        return self._players

    @property
    def periods(self) -> Union[pl.DataFrame, "SparkDataFrame"]:
        """Get periods DataFrame with period_id, start_frame_id, end_frame_id."""
        return self._periods

    def collect(self) -> Union[pl.DataFrame, "SparkDataFrame"]:
        """Collect the tracking LazyFrame into a DataFrame.

        Convenience method that calls self.tracking.collect().
        For Polars lazy datasets, this triggers the actual data loading.
        For PySpark, returns the DataFrame as-is (PySpark is inherently lazy).

        After calling collect(), subsequent access to dataset.tracking
        will return the collected DataFrame instead of the LazyFrame.

        Returns
        -------
        pl.DataFrame or pyspark.sql.DataFrame
            The collected tracking data.

        Examples
        --------
        >>> dataset = tracab.load_tracking("raw.dat", "meta.xml")
        >>> df = dataset.collect()  # Same as dataset.tracking.collect()
        >>> dataset.tracking  # Now returns DataFrame, not LazyFrame
        """
        # For PySpark, just return the DataFrame (it's inherently lazy)
        if self._engine == "pyspark":
            return self._tracking

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
            _engine=self._engine,
            _provider=self._provider,
            _cache_key=self._cache_key,
        )

    def to_polars(self) -> "TrackingDataset":
        """Convert all DataFrames to Polars.

        If already using Polars engine, returns self unchanged.
        For PySpark DataFrames, converts via Arrow/pandas interchange.

        Returns
        -------
        TrackingDataset
            New TrackingDataset with all Polars DataFrames,
            or self if already Polars.

        Examples
        --------
        >>> dataset_spark = secondspectrum.load_tracking(..., engine="pyspark")
        >>> dataset_polars = dataset_spark.to_polars()
        >>> dataset_polars.engine
        'polars'
        """
        if self._engine == "polars":
            return self

        from fastforward._engine import spark_to_polars

        return TrackingDataset(
            tracking=spark_to_polars(self._tracking),
            metadata=spark_to_polars(self._metadata),
            teams=spark_to_polars(self._teams),
            players=spark_to_polars(self._players),
            periods=spark_to_polars(self._periods),
            _engine="polars",
            _provider=self._provider,
            _cache_key=self._cache_key,
        )

    def to_pyspark(
        self, spark: Optional["SparkSession"] = None
    ) -> "TrackingDataset":
        """Convert all DataFrames to PySpark.

        If already using PySpark engine, returns self unchanged.
        For Polars DataFrames, converts via Arrow interchange.

        Parameters
        ----------
        spark : SparkSession, optional
            SparkSession to use. If None, gets or creates one.

        Returns
        -------
        TrackingDataset
            New TrackingDataset with all PySpark DataFrames,
            or self if already PySpark.

        Examples
        --------
        >>> dataset_polars = secondspectrum.load_tracking(...)
        >>> dataset_spark = dataset_polars.to_pyspark()
        >>> dataset_spark.engine
        'pyspark'
        """
        if self._engine == "pyspark":
            return self

        from fastforward._engine import polars_to_spark, get_spark_session

        if spark is None:
            spark = get_spark_session()

        # For Polars LazyFrame, collect first
        tracking = self._tracking
        if isinstance(tracking, pl.LazyFrame):
            tracking = tracking.collect()

        return TrackingDataset(
            tracking=polars_to_spark(tracking, spark),
            metadata=polars_to_spark(self._metadata, spark),
            teams=polars_to_spark(self._teams, spark),
            players=polars_to_spark(self._players, spark),
            periods=polars_to_spark(self._periods, spark),
            _engine="pyspark",
            _provider=self._provider,
            _cache_key=self._cache_key,
        )

    def write_cache(self) -> None:
        """Write tracking data to cache.

        Caches the original (unmodified) tracking data to the global cache directory.
        The cache directory is configured via fastforward.set_cache_dir() or
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
        from fastforward._cache import (
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

    # =========================================================================
    # Transformation state properties (read from private attributes)
    # =========================================================================

    @property
    def coordinate_system(self) -> str:
        """Get the current coordinate system."""
        return self._coordinate_system

    @property
    def orientation(self) -> str:
        """Get the current orientation."""
        return self._orientation

    @property
    def pitch_dimensions(self) -> Tuple[float, float]:
        """Get current pitch dimensions (length, width) in meters."""
        return (self._pitch_length, self._pitch_width)

    # =========================================================================
    # Transform method
    # =========================================================================

    def transform(
        self,
        to_orientation: Optional[str] = None,
        to_dimensions: Optional[Tuple[float, float]] = None,
        to_coordinates: Optional[str] = None,
    ) -> "TrackingDataset":
        """Transform tracking data to different orientation, dimensions, and/or coordinates.

        Transformations are applied in the correct order internally:
        1. Orientation (flip) - while in CDF/meters
        2. Dimensions (zone-based scaling) - while in CDF/meters
        3. Coordinates (unit/origin conversion) - last step

        Parameters
        ----------
        to_orientation : str, optional
            Target orientation. Options include:
            - "static_home_away": Home team attacks left-to-right in both halves
            - "static_away_home": Away team attacks left-to-right in both halves
            Note: Orientation transforms flip x and y around the center.
        to_dimensions : tuple of (float, float), optional
            Target pitch dimensions (length, width) in meters.
            Uses zone-based scaling to preserve IFAB pitch feature proportions.
        to_coordinates : str, optional
            Target coordinate system. Options include:
            - "cdf": Center origin, meters (default)
            - "tracab": Center origin, centimeters
            - "opta": Bottom-left origin, 0-100 scale
            - "kloppy": Top-left origin, 0-1 scale
            - "sportvu": Top-left origin, meters

        Returns
        -------
        TrackingDataset
            New dataset with transformed data, or self if no changes needed.

        Examples
        --------
        >>> dataset = secondspectrum.load_tracking("tracking.jsonl", "meta.json")
        >>> # Single transformation
        >>> tracab = dataset.transform(to_coordinates="tracab")
        >>> # Multiple transformations (order handled internally)
        >>> result = dataset.transform(
        ...     to_orientation="static_away_home",
        ...     to_dimensions=(105.0, 68.0),
        ...     to_coordinates="tracab",
        ... )
        """
        if self._engine != "polars":
            raise NotImplementedError(
                "transform() is currently only supported for Polars DataFrames"
            )

        from fastforward._transforms import (
            transform_coordinates as _coord,
            transform_dimensions as _dim,
            transform_orientation as _orient,
        )

        # Get current state from properties (which read from metadata)
        current_orientation = self.orientation
        current_coord_system = self.coordinate_system
        current_length, current_width = self.pitch_dimensions

        # Check if any transformation is needed
        needs_orientation = (
            to_orientation is not None and to_orientation != current_orientation
        )
        needs_dimensions = (
            to_dimensions is not None
            and not (
                abs(current_length - to_dimensions[0]) < 0.001
                and abs(current_width - to_dimensions[1]) < 0.001
            )
        )
        needs_coordinates = (
            to_coordinates is not None and to_coordinates != current_coord_system
        )

        if not (needs_orientation or needs_dimensions or needs_coordinates):
            return self  # No changes needed

        # Collect if lazy
        tracking = self._tracking
        if isinstance(tracking, pl.LazyFrame):
            tracking = tracking.collect()

        # Track new state
        new_orientation = to_orientation if needs_orientation else current_orientation
        new_length = to_dimensions[0] if needs_dimensions else current_length
        new_width = to_dimensions[1] if needs_dimensions else current_width
        new_coord_system = to_coordinates if needs_coordinates else current_coord_system

        # Current state for transformations
        curr_coord_system = current_coord_system
        curr_length = current_length
        curr_width = current_width

        # Step 0: If not in CDF and we need orientation or dimension changes, convert to CDF first
        if curr_coord_system != "cdf" and (needs_orientation or needs_dimensions):
            tracking = _coord(tracking, curr_coord_system, "cdf", curr_length, curr_width)
            curr_coord_system = "cdf"

        # Step 1: Apply orientation transformation (in CDF)
        # Determine if we need to flip based on orientation change
        if needs_orientation:
            # Static orientations: flip if switching between home_away and away_home
            current_is_home_away = "home_away" in current_orientation
            target_is_home_away = "home_away" in to_orientation
            flip = current_is_home_away != target_is_home_away
            tracking = _orient(tracking, flip)

        # Step 2: Apply dimension transformation (in CDF)
        if needs_dimensions:
            tracking = _dim(
                tracking,
                curr_length,
                curr_width,
                to_dimensions[0],
                to_dimensions[1],
            )
            curr_length, curr_width = to_dimensions

        # Step 3: Apply coordinate system transformation (last)
        if curr_coord_system != new_coord_system:
            tracking = _coord(
                tracking, curr_coord_system, new_coord_system, curr_length, curr_width
            )

        # Update metadata DataFrame with new transformation state
        new_metadata = self._metadata.with_columns([
            pl.lit(new_coord_system).alias("coordinate_system"),
            pl.lit(new_orientation).alias("orientation"),
            pl.lit(new_length).cast(pl.Float32).alias("pitch_length"),
            pl.lit(new_width).cast(pl.Float32).alias("pitch_width"),
        ])

        return TrackingDataset(
            tracking=tracking,
            metadata=new_metadata,
            teams=self._teams,
            players=self._players,
            periods=self._periods,
            _engine=self._engine,
            _provider=self._provider,
            _cache_key=self._cache_key,
            _coordinate_system=new_coord_system,
            _orientation=new_orientation,
            _pitch_length=new_length,
            _pitch_width=new_width,
        )

    def __repr__(self) -> str:
        """String representation."""
        tracking_type = type(self._tracking).__name__

        # Handle game_id extraction for both Polars and PySpark
        if self._engine == "pyspark":
            try:
                row = self._metadata.first()
                game_id = row["game_id"] if row and "game_id" in self._metadata.columns else "unknown"
            except Exception:
                game_id = "unknown"
            n_periods = self._periods.count()
            n_players = self._players.count()
        else:
            game_id = self._metadata["game_id"][0] if "game_id" in self._metadata.columns else "unknown"
            n_periods = len(self._periods)
            n_players = len(self._players)

        return (
            f"TrackingDataset(\n"
            f"  engine={self._engine!r},\n"
            f"  game_id={game_id!r},\n"
            f"  tracking={tracking_type},\n"
            f"  periods={n_periods},\n"
            f"  players={n_players}\n"
            f")"
        )
