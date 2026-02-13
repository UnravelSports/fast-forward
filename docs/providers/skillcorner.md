# SkillCorner

Load tracking data from **SkillCorner** format.

## Function Signature

```python
from fastforward import skillcorner

dataset = skillcorner.load_tracking(
    raw_data="tracking_extrapolated.jsonl",
    meta_data="match.json",
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raw_data` | FileLike | *required* | Path to JSONL tracking file |
| `meta_data` | FileLike | *required* | Path to JSON match file |
| `layout` | str | `"long"` | `"long"`, `"long_ball"`, or `"wide"` |
| `coordinates` | str | `"cdf"` | Target coordinate system |
| `orientation` | str | `"static_home_away"` | Target orientation |
| `only_alive` | bool | `True` | Only include frames where ball is in play |
| `include_empty_frames` | bool | `False` | Include frames with no detected players |
| `include_game_id` | bool \| str | `True` | Add game_id column |
| `engine` | str | `"polars"` | `"polars"` or `"pyspark"` |

## File Format

**Tracking data** (JSONL): Typically named `tracking_extrapolated.jsonl`. One JSON object per frame.

**Metadata** (JSON): Typically named `match.json`. Contains match info, teams, and players.

## Example

```python
from fastforward import skillcorner

dataset = skillcorner.load_tracking(
    raw_data="tracking_extrapolated.jsonl",
    meta_data="match.json",
    include_empty_frames=False,
)

print(dataset.tracking.shape)
print(dataset.players)
```

## Notes

- SkillCorner uses CDF-compatible coordinates natively (center origin, meters)
- Empty frames (frames with no detected players) are excluded by default. Set `include_empty_frames=True` to include them
