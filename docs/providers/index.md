# Providers

fast-forward supports 10 tracking data providers. Each provider has a `load_tracking()` function that returns a [`TrackingDataset`](../concepts/dataset.md).

## Comparison Table

| Provider | Files | Format | Native Coords | Layouts | Special Parameters |
|----------|-------|--------|---------------|---------|-------------------|
| [CDF](cdf.md) | 2 | JSONL + JSON | CDF | long, long_ball, wide | `exclude_missing_ball_frames` |
| [GradientSports](gradientsports.md) | 3 | JSONL + JSON | CDF | long, long_ball, wide | `include_incomplete_frames`, `roster_data` |
| [HawkEye](hawkeye.md) | Multi | Text + JSON/XML | CDF | long | `pitch_length`, `pitch_width`, `object_id`, `include_officials`, `parallel` |
| [Respovision](respovision.md) | 1 | JSONL | Sportec Event | long, long_ball, wide | `pitch_length`, `pitch_width`, `include_joint_angles`, `include_officials` |
| [SecondSpectrum](secondspectrum.md) | 2 | JSONL + JSON | CDF | long, long_ball, wide | `exclude_missing_ball_frames` |
| [Signality](signality.md) | 3+ | JSON | CDF | long, long_ball, wide | `include_officials`, `parallel`, `venue_information` |
| [SkillCorner](skillcorner.md) | 2 | JSONL + JSON | CDF | long, long_ball, wide | `include_empty_frames` |
| [Sportec](sportec.md) | 2 | XML | CDF | long, long_ball, wide | `include_officials` |
| [StatsPerform](statsperform.md) | 2 | MA25 + MA1 | SportVU | long, long_ball, wide | `pitch_length`, `pitch_width`, `include_officials` |
| [Tracab](tracab.md) | 2 | DAT/JSON + XML/JSON | Tracab (cm) | long, long_ball, wide | — |

## Common Parameters

All providers share these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout` | str | `"long"` | DataFrame layout: `"long"`, `"long_ball"`, or `"wide"` |
| `coordinates` | str | `"cdf"` | Target [coordinate system](../concepts/coordinate-systems.md) |
| `orientation` | str | `"static_home_away"` | Target [orientation](../concepts/orientations.md) |
| `only_alive` | bool | `True` | Only include frames where ball is in play |
| `include_game_id` | bool \| str | `True` | Add `game_id` column (`True` uses provider default, or pass a custom string) |
| `engine` | str | `"polars"` | DataFrame engine: `"polars"` or `"pyspark"` |
| `spark_session` | SparkSession | `None` | PySpark session (only needed if `engine="pyspark"`) |

## Usage Pattern

```python
from fastforward import secondspectrum  # or any provider

dataset = secondspectrum.load_tracking(
    raw_data="tracking.jsonl",
    meta_data="metadata.json",
    layout="long",
    coordinates="cdf",
    orientation="static_home_away",
    only_alive=True,
)

# All providers return a TrackingDataset
dataset.tracking    # pl.DataFrame
dataset.metadata    # pl.DataFrame
dataset.teams       # pl.DataFrame
dataset.players     # pl.DataFrame
dataset.periods     # pl.DataFrame
```
