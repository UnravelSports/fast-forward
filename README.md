# fast-forward

Fast tracking data loader using Rust. Supports multiple providers: SecondSpectrum, SkillCorner, Sportec, Tracab, HawkEye, CDF, GradientSports, Signality, and StatsPerform.

## Installation

```bash
pip install fast-forward
```

## Usage

```python
from fastforward import secondspectrum, skillcorner, sportec, tracab, hawkeye, cdf, gradientsports, signality, statsperform

# Load tracking data (example with SecondSpectrum)
dataset = secondspectrum.load_tracking(
    raw_data="tracking.jsonl",
    meta_data="metadata.json",
    layout="long",           # "long", "long_ball", "wide"
    coordinates="cdf",       # Coordinate system
    orientation="static_home_away",
    only_alive=True,
    lazy=False,
)

# Access data via properties
tracking_df = dataset.tracking    # pl.DataFrame or pl.LazyFrame
metadata_df = dataset.metadata    # Match metadata
teams_df = dataset.teams          # Team info
players_df = dataset.players      # Player info
periods_df = dataset.periods      # Period info
```
