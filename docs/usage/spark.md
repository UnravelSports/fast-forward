# Distributed Compute

Pick the engine by what you're plugging into:

- For **Spark** (`mapInArrow`, `createDataFrame`) and **Ray**
  (`map_batches`): `engine="arrow[spark]"`. Both reject Polars's
  `string_view` and `duration[ms]` Arrow types; `arrow[spark]` pre-casts
  them to `string` and `int64`.
- For **Dask** (`.to_pandas()` in the worker) or bare pyarrow: `engine="arrow"`
  is fine. Pandas absorbs `string_view` and `duration[ms]` directly.

Both engines return the same data, just with different Arrow dtypes. Both
accept raw `bytes` (or `bytearray`, `memoryview`, `io.BytesIO`) for the
tracking and metadata inputs; neither accepts file paths or URLs. Neither
imports kloppy on executors, so workers stay small.

## What `fast-forward` owns vs what your framework owns

| Concern                                  | fast-forward                  | Framework (Spark/Dask/Ray)       |
| ---------------------------------------- | ----------------------------- | -------------------------------- |
| Parsing one match very fast              | Yes                           |                                  |
| Coordinate / orientation / layout        | Yes                           |                                  |
| Schema derivation (drift-free)           | Yes                           |                                  |
| Listing matches in object storage        |                               | Yes                              |
| Distributing work across workers         |                               | Yes                              |
| Persistence (Parquet / Iceberg / etc.)   |                               | Yes                              |
| Dedup / streaming / checkpoints          |                               | Yes                              |

That separation is the whole point. The library does not need to know about
Spark, Dask, or Ray. None of them needs to know about the library. They meet
at the Arrow boundary, briefly, and otherwise leave each other alone.

## Schema helpers

Two ways to get the Arrow / PySpark schemas, both producing the same
`Schemas` namespace with 10 lazy properties (5 tables x {Arrow, PySpark}):

| Table       | Arrow (`pyarrow.Schema`) | PySpark (`StructType`)   |
| ----------- | ------------------------ | ------------------------ |
| `tracking`  | `s.tracking`             | `s.tracking_spark`       |
| `metadata`  | `s.metadata`             | `s.metadata_spark`       |
| `teams`     | `s.teams`                | `s.teams_spark`          |
| `players`   | `s.players`              | `s.players_spark`        |
| `periods`   | `s.periods`              | `s.periods_spark`        |

Schemas are derived from the Rust parser, so they cannot drift from what
`load_tracking` actually produces.

Two ways to reach this namespace:

- `dataset.schemas` is available on any loaded dataset, with the load's
  kwargs already bound. Use it when you have a sample match to load on the
  driver.
- `skillcorner.schemas(...)` is for when you need the schema before any data
  is loaded (for example, declaring a Spark `mapInArrow` output schema
  before any worker runs). It accepts the same kwargs as `load_tracking`,
  so unpack a single config dict into both.

```python
LOAD_KWARGS = dict(
    engine="arrow[spark]",
    layout="long",
    include_game_id=True,
    include_ball_owning_player=True,
    include_is_detected=True,
)

out_schema = skillcorner.schemas(**LOAD_KWARGS).tracking_spark
```

### What `engine=` controls in `schemas()`

The same four values `load_tracking` accepts work on `schemas()`:

- `engine="polars"` (default) or `engine="arrow"`: Arrow schemas use
  Polars-style types (`string_view`, `duration[ms]`).
- `engine="pyspark"` or `engine="arrow[spark]"`: Arrow schemas use
  Spark-compat types (`string`, `int64`).

`*_spark` properties are always Spark-compatible regardless of `engine`.

### Wide layout

`layout="wide"` has per-game column names (player IDs become column names),
so `s.tracking` and `s.tracking_spark` raise `NotImplementedError`. The other
four schemas still work.

## Examples

Three frameworks, the same shape: a DataFrame with one row per match
(carrying the raw bytes), a worker function that calls `load_tracking`, and
the framework that distributes the work. The fast-forward call is identical
across all three; only the framework glue changes.

### Spark `mapInArrow`

You give Spark a DataFrame with one row per match (containing the raw bytes
of both files). Spark distributes the rows across workers. Each worker calls
`fast-forward` inside the UDF and yields the parsed rows back.

Define the load kwargs once and reuse them for both the schema declaration
and the per-worker load. They must agree; the dict-unpack pattern below
makes that agreement structural rather than a thing to remember.

```python
from pyspark.sql import SparkSession

from fastforward import skillcorner

spark = SparkSession.builder.getOrCreate()

# Define every kwarg once. Both the schema declaration and the worker load
# unpack the same dict, so they cannot drift.
LOAD_KWARGS = dict(
    engine="arrow[spark]",
    layout="long",
    include_game_id=True,
    include_ball_owning_player=True,
    include_is_detected=True,
)

# 1. Build a Spark DataFrame with one row per match.
#    Required columns:
#      match_id:       string
#      tracking_bytes: binary
#      meta_bytes:     binary
#    How you construct it is up to you. Common patterns:
#      spark.read.format("binaryFile") for files in object storage
#      a UDF that fetches bytes from S3 / GCS / Azure Blob
#      joining two upstream tables that already carry the bytes columns
matches_df = ...   # build however fits your environment

# 2. Declare the output schema upfront. Same kwargs as the per-worker load.
out_schema = skillcorner.schemas(**LOAD_KWARGS).tracking_spark

# 3. The worker UDF. Runs once per Spark partition.
def parse_skillcorner_match_udf(iterator):
    """Each batch is a pyarrow.RecordBatch with (match_id, tracking_bytes,
    meta_bytes) columns. For each row, parse one match and yield the tracking
    rows back.
    """
    for batch in iterator:
        tracking_col = batch.column("tracking_bytes").to_pylist()
        meta_col = batch.column("meta_bytes").to_pylist()
        for raw, meta in zip(tracking_col, meta_col):
            dataset = skillcorner.load_tracking(raw, meta, **LOAD_KWARGS)
            for record_batch in dataset.tracking.to_batches():
                yield record_batch

# 4. Spark distributes one match per task across the cluster.
tracking_df = matches_df.mapInArrow(parse_skillcorner_match_udf, schema=out_schema)

# 5. Write to wherever you prefer: Parquet, Iceberg, Delta, JDBC, etc.
tracking_df.write.mode("append").parquet("s3a://my-bucket/skillcorner-tracking/")
```

`mapInArrow` is the key piece. Available in open-source PySpark since 3.3,
works on every distribution. Spark distributes the Arrow batches across the
cluster automatically; the driver does only coordination.

### Ray `map_batches`

Ray's internal pyarrow operations don't yet support `string_view`, so use
`engine="arrow[spark]"` (same as Spark).

```python
import ray
import pyarrow as pa

from fastforward import skillcorner

# Build a Ray Dataset with one row per match.
# Required columns:
#   match_id:       string
#   tracking_bytes: binary
#   meta_bytes:     binary
# Common patterns:
#   ray.data.from_items([{...}, ...]) for in-memory lists
#   ray.data.read_binary_files(...) for files in object storage
#   ray.data.from_pandas(df) when you already have a pandas frame
matches_ds = ...   # build however fits your environment

def parse_batch(batch):
    out = []
    for raw, meta in zip(batch["tracking_bytes"], batch["meta_bytes"]):
        ds = skillcorner.load_tracking(bytes(raw), bytes(meta), engine="arrow[spark]")
        out.append(ds.tracking)
    return pa.concat_tables(out)

tracking_ds = matches_ds.map_batches(parse_batch, batch_format="pyarrow")
```

### Dask `map_partitions`

Dask consumes Polars-style Arrow types via `.to_pandas()` natively. Use
`engine="arrow"` (no Spark normalization).

```python
import pandas as pd
import dask.dataframe as dd

from fastforward import skillcorner

# Build a Dask DataFrame with one row per match.
# Required columns:
#   match_id:       string
#   tracking_bytes: bytes (object dtype)
#   meta_bytes:     bytes (object dtype)
# Common patterns:
#   dd.from_pandas(df, npartitions=N) for an existing pandas frame
#   dd.read_parquet(...) when the bytes already sit in Parquet
matches_ddf = ...   # build however fits your environment

def parse_partition(df):
    out = []
    for _, row in df.iterrows():
        ds = skillcorner.load_tracking(
            row["tracking_bytes"], row["meta_bytes"],
            engine="arrow",
        )
        out.append(ds.tracking.to_pandas())
    return pd.concat(out, ignore_index=True)

# meta describes the output frame's columns and dtypes; build it from the
# same kwargs the worker uses to keep the contract drift-free.
tracking_meta = (
    skillcorner.schemas(engine="arrow").tracking.empty_table().to_pandas()
)
tracking_ddf = matches_ddf.map_partitions(parse_partition, meta=tracking_meta)
```

## Actual produced schema

The column-by-column dtypes for the four engine values are documented on the [TrackingDataset Per-engine schema](../concepts/dataset.md#per-engine-schema) page. The Gotchas section below lists the surprises specific to Spark pipelines.

Notes on conditional and nullable columns (apply to all engines):

- `game_id`, `ball_owning_player_id`, and `is_detected` are present only when their respective `include_*` flag is `True`.
- `ball_owning_team_id` and `ball_owning_player_id` are nullable (null when SkillCorner did not record possession on that frame).
- `ball_state` is one of `"alive"` or `"dead"`.

## Gotchas

- **`timestamp` is `int64` milliseconds in Spark**, not a `float` seconds. Polars
  emits it as a `Duration(ms)`, which Spark's Arrow path does not accept.
  `engine="arrow[spark]"` casts to `int64` rather than guess at unit
  conversions you might not want. Divide by 1000 yourself if you need seconds.
- **`frame_id` is `LongType` (int64) in Spark**, not `IntegerType`. The
  parser's natural type is `UInt32`, which PySpark cannot represent in its
  Arrow path; we cast up to `Int64` rather than risk overflow.
- **Do not hand-write the `StructType`.** Call
  `skillcorner.schemas(...).tracking_spark` so your code keeps working when
  the schema evolves.
- **Wide layout is not supported with the arrow engines.** `mapInArrow`
  requires a static output schema, and `layout="wide"` has per-game column
  names (player IDs become column names). Use `layout="long"` or
  `layout="long_ball"`. Passing `layout="wide"` raises `NotImplementedError`
  from both `load_tracking` and `schemas().tracking`.

## Databricks Runtime compatibility

`engine="arrow"` and `engine="arrow[spark]"` require pyarrow >= 14.
Databricks Runtime 15+ ships pyarrow 14; DBR 14 LTS and older ship
pyarrow 8-12.

To use the arrow engines on DBR <= 14, install a newer pyarrow on the cluster:

```python
%pip install pyarrow>=14
```
