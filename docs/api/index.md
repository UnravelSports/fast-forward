# API Reference

## TrackingDataset

::: kloppy_light._dataset.TrackingDataset
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

::: kloppy_light.cdf.load_tracking
    options:
      show_root_heading: false

### SecondSpectrum

::: kloppy_light.secondspectrum.load_tracking
    options:
      show_root_heading: false

### SkillCorner

::: kloppy_light.skillcorner.load_tracking
    options:
      show_root_heading: false

### Sportec

::: kloppy_light.sportec.load_tracking
    options:
      show_root_heading: false

### Tracab

::: kloppy_light.tracab.load_tracking
    options:
      show_root_heading: false

### HawkEye

::: kloppy_light.hawkeye.load_tracking
    options:
      show_root_heading: false

### Signality

::: kloppy_light.signality.load_tracking
    options:
      show_root_heading: false

### StatsPerform

::: kloppy_light.statsperform.load_tracking
    options:
      show_root_heading: false

### GradientSports

::: kloppy_light.gradientsports.load_tracking
    options:
      show_root_heading: false

### Respovision

::: kloppy_light.respovision.load_tracking
    options:
      show_root_heading: false

## Transforms

### transform_coordinates

::: kloppy_light._transforms.transform_coordinates
    options:
      show_root_heading: false

### transform_dimensions

::: kloppy_light._transforms.transform_dimensions
    options:
      show_root_heading: false

### transform_orientation

::: kloppy_light._transforms.transform_orientation
    options:
      show_root_heading: false
