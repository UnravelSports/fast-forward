# kloppy-light

Fast SecondSpectrum tracking data loader using Rust.

## Installation

```bash
pip install kloppy-light
```

## Usage

```python
from kloppy_light import secondspectrum

tracking_df, team_df, player_df = secondspectrum.load_tracking(
    "tracking.jsonl",
    "metadata.json",
    layout="long",
    coordinates="cdf"
)
```
