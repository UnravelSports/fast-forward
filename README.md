<p align="center">
  <img src="docs/assets/logos/fastforward-gradient-logo.png" alt="fast-forward logo" width="400">
</p>

[![Python](<https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue>)](https://pypi.org/project/fast-forward-football/)

## Fast, Robust, Rust-powered Positional Tracking Data Loading for Football Analytics

FAST<i>FORWARD</i> Supports 12 positional tracking data providers: SecondSpectrum, SkillCorner, Sportec, Tracab, HawkEye, GradientSports, Signality, StatsPerform, RespoVision, OptaVision, SciSports. Additionally, it supports the [Common Data Format](https://www.cdf.football).

This project owes a depth of gratitude to [Kloppy](https://kloppy.pysport.org/) and all its contributors.

> ⚠️ **STATUS**: This project is currently in Beta. Only the Python bindings are available (not the Rust code). Please try it, and report any issues [here](https://github.com/UnravelSports/fast-forward/issues).

> ❓ **ENQUIRIES**: If you work for a (skeletal) tracking provider, or have access to (skeletal) tracking data from an unsupported provider and would like to see support for your data, please contact me at joris at unravelsports dot com.

## Installation

```bash
pip install fast-forward-football
```

## Usage

```python
from fastforward import secondspectrum, skillcorner, sportec, tracab, hawkeye, cdf, gradientsports, signality, statsperform, respovision, optavision, scisports

# Load tracking data (example with SecondSpectrum)
dataset = secondspectrum.load_tracking(
    raw_data="tracking.jsonl",
    meta_data="metadata.json",
    layout="long",           # "long", "long_ball", "wide"
    coordinates="cdf",       # Coordinate system
    orientation="static_home_away",
    only_alive=True,
)

# Access data via properties
tracking_df = dataset.tracking    # pl.DataFrame 
metadata_df = dataset.metadata    # Match metadata
teams_df = dataset.teams          # Team info
players_df = dataset.players      # Player info
periods_df = dataset.periods      # Period info
```

## Provider Support

| Provider                         | Tracking Data |      Public Data      |           Docs           | Notes                                              |
| -------------------------------- | :-----------: | :--------------------: | :----------------------: | :------------------------------------------------- |
| [CDF][cdf]                       |      ✓      |                        |      [↗][cdf-doc]      |                                                    |
| [GradientSports][gradientsports] |      ✓      |     [↗][pff-data]     | [↗][gradientsports-doc] | Formerly PFF                                       |
| [Hawkeye (2D)][hawkeye]          |      ✓      |                        |    [↗][hawkeye-doc]    | Joint tracking data is not yet supported           |
| [OptaVision][optavision]         |      ✓      |                        |   [↗][optavision-doc]   |                                                    |
| [RespoVision][respovision]       |      ✓      |                        |  [↗][respovision-doc]  | Includes support for v1 data                       |
| [SciSports][scisports]           |      ✓      |                        |   [↗][scisports-doc]   |                                                    |
| [SecondSpectrum][ss]             |      ✓      |                        |       [↗][ss-doc]       |                                                    |
| [Signality][signality]           |      ✓      |                        |   [↗][signality-doc]   |                                                    |
| [SkillCorner][skillcorner]       |      ✓      | [↗][skillcorner-data] |  [↗][skillcorner-doc]  | Includes support for v3 data                       |
| [Sportec][sportec]               |      ✓      |   [↗][sportec-data]   |    [↗][sportec-doc]    |                                                    |
| [Stats Perform][statsperform]    |      ✓      |                        |  [↗][statsperform-doc]  | Includes support for MA1, MA3, and MA25 data feeds |
| [Tracab][tracab]                 |      ✓      |                        |     [↗][tracab-doc]     |                                                    |

## Distributed Compute

For Spark, Ray, and Dask workflows, fast-forward exposes `engine="arrow"` and `engine="arrow[spark]"` paths that return `pyarrow.Table` directly without importing kloppy on executors. See [Distributed Compute](docs/concepts/distributed-compute.md).

## Benchmarks

![Load Time](https://raw.githubusercontent.com/UnravelSports/fast-forward/main/docs/assets/images/benchmark_load_time.png)

[cdf]: https://www.cdf.football
[cdf-doc]: https://fast-forward.readthedocs.io/en/latest/providers/cdf/
[gradientsports]: https://www.gradientsports.com/
[gradientsports-doc]: https://fast-forward.readthedocs.io/en/latest/providers/gradientsports/
[hawkeye]: https://www.hawkeyeinnovations.com/data
[hawkeye-doc]: https://fast-forward.readthedocs.io/en/latest/providers/hawkeye/
[optavision]: https://www.statsperform.com/opta-vision/
[optavision-doc]: https://fast-forward.readthedocs.io/en/latest/providers/optavision/
[pff-data]: https://drive.google.com/drive/u/0/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa
[respovision]: https://respo.vision/
[respovision-doc]: https://fast-forward.readthedocs.io/en/latest/providers/respovision/
[scisports]: https://www.scisports.com/
[scisports-doc]: https://fast-forward.readthedocs.io/en/latest/providers/scisports/
[signality]: https://www.spiideo.com/
[signality-doc]: https://fast-forward.readthedocs.io/en/latest/providers/signality/
[skillcorner]: https://skillcorner.com/
[skillcorner-data]: https://github.com/SkillCorner/opendata
[skillcorner-doc]: https://fast-forward.readthedocs.io/en/latest/providers/skillcorner/
[sportec]: https://sportec-solutions.de/en/index.html
[sportec-data]: https://www.nature.com/articles/s41597-025-04505-y
[sportec-doc]: https://fast-forward.readthedocs.io/en/latest/providers/sportec/
[ss]: https://www.geniussports.com/
[ss-doc]: https://fast-forward.readthedocs.io/en/latest/providers/secondspectrum/
[statsperform]: https://www.statsperform.com/
[statsperform-doc]: https://fast-forward.readthedocs.io/en/latest/providers/statsperform/
[tracab]: https://tracab.com/products/tracab-technologies/
[tracab-doc]: https://fast-forward.readthedocs.io/en/latest/providers/tracab/
