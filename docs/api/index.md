# API Reference

## TrackingDataset

::: fastforward._dataset.TrackingDataset
    options:
      show_source: false
      show_root_heading: false
      show_signature_annotations: false
      members:
        - tracking
        - metadata
        - teams
        - players
        - periods
        - engine
        - coordinate_system
        - orientation
        - pitch_dimensions
        - to_polars
        - to_pyspark
        - transform

## Providers

Each provider module exposes a `load_tracking()` function that returns a `TrackingDataset`.

### CDF

::: fastforward.cdf.load_tracking
    options:
      show_root_heading: false

### SecondSpectrum

::: fastforward.secondspectrum.load_tracking
    options:
      show_root_heading: false

### SkillCorner

::: fastforward.skillcorner.load_tracking
    options:
      show_root_heading: false

### Sportec

::: fastforward.sportec.load_tracking
    options:
      show_root_heading: false

### Tracab

::: fastforward.tracab.load_tracking
    options:
      show_root_heading: false

### HawkEye

::: fastforward.hawkeye.load_tracking
    options:
      show_root_heading: false

### Signality

::: fastforward.signality.load_tracking
    options:
      show_root_heading: false

### StatsPerform

::: fastforward.statsperform.load_tracking
    options:
      show_root_heading: false

### GradientSports

::: fastforward.gradientsports.load_tracking
    options:
      show_root_heading: false

### Respovision

::: fastforward.respovision.load_tracking
    options:
      show_root_heading: false

## Transforms

### transform_coordinates

::: fastforward._transforms.transform_coordinates
    options:
      show_root_heading: false

### transform_dimensions

::: fastforward._transforms.transform_dimensions
    options:
      show_root_heading: false

### transform_orientation

::: fastforward._transforms.transform_orientation
    options:
      show_root_heading: false
