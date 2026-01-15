"""Engine abstraction for DataFrame backends.

This module provides utilities for converting between Polars and PySpark DataFrames,
enabling kloppy-light to support multiple DataFrame engines.
"""

from typing import TYPE_CHECKING, Literal, Optional

import polars as pl

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession

Engine = Literal["polars", "pyspark"]


def validate_engine(engine: str) -> Engine:
    """Validate engine parameter.

    Parameters
    ----------
    engine : str
        Engine name to validate

    Returns
    -------
    Engine
        Validated engine literal

    Raises
    ------
    ValueError
        If engine is not 'polars' or 'pyspark'
    """
    if engine not in ("polars", "pyspark"):
        raise ValueError(
            f"Invalid engine: {engine!r}. Must be 'polars' or 'pyspark'."
        )
    return engine  # type: ignore[return-value]


def get_spark_session() -> "SparkSession":
    """Get or create a SparkSession.

    Returns
    -------
    SparkSession
        Active or newly created SparkSession

    Raises
    ------
    ImportError
        If PySpark is not installed
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "PySpark is required for engine='pyspark'. "
            "Install with: pip install kloppy-light[pyspark]"
        )

    return SparkSession.builder.getOrCreate()


def polars_to_spark(
    df: pl.DataFrame, spark: Optional["SparkSession"] = None
) -> "SparkDataFrame":
    """Convert a Polars DataFrame to a PySpark DataFrame.

    Uses Apache Arrow as the interchange format for efficient conversion.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame to convert
    spark : SparkSession, optional
        SparkSession to use. If None, gets or creates one.

    Returns
    -------
    SparkDataFrame
        PySpark DataFrame with the same data
    """
    if spark is None:
        spark = get_spark_session()

    # Convert via Arrow -> pandas -> Spark
    # This path is optimized when spark.sql.execution.arrow.pyspark.enabled=true
    arrow_table = df.to_arrow()
    return spark.createDataFrame(arrow_table.to_pandas())


def spark_to_polars(df: "SparkDataFrame") -> pl.DataFrame:
    """Convert a PySpark DataFrame to a Polars DataFrame.

    Uses pandas as an intermediate format with Arrow optimization.

    Parameters
    ----------
    df : SparkDataFrame
        PySpark DataFrame to convert

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with the same data
    """
    # Convert via Spark -> pandas -> Polars
    # Arrow optimization is used when available
    pandas_df = df.toPandas()
    return pl.from_pandas(pandas_df)
