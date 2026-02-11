# Transformations

The `TrackingDataset.transform()` method lets you change the coordinate system, orientation, or pitch dimensions of your tracking data after loading.

## Basic Usage

```python
transformed = dataset.transform(
    to_coordinates="opta",
    to_orientation="home_away",
    to_dimensions=(100, 100),
)
```

All three parameters are optional — you can transform any combination.

## Transform Order

When multiple transforms are applied in a single call, they are always executed in this order:

1. **Orientation** — flip coordinates if needed
2. **Dimensions** — scale to new pitch dimensions
3. **Coordinates** — convert between coordinate systems

This order is enforced internally to ensure correct results.

## Coordinate Transforms

Transform between any of the [supported coordinate systems](coordinate-systems.md):

```python
# Tracab (centimeters, center origin) -> Opta (0-100, bottom-left)
dataset = dataset.transform(to_coordinates="opta")

# Check new state
print(dataset.coordinate_system)  # "opta"
```

All coordinate transformations use **CDF as an intermediate**: source -> CDF -> target. This means any pair of coordinate systems can be converted.

## Orientation Transforms

Change the attacking direction convention:

```python
# Static -> alternating
dataset = dataset.transform(to_orientation="home_away")

print(dataset.orientation)  # "home_away"
```

When orientation requires flipping, both x and y coordinates are **negated** (reflected around the center origin). See [Orientations](orientations.md) for details on each option.

## Dimension Transforms

Scale coordinates to different pitch dimensions:

```python
# Original: 105m x 68m -> Target: 100m x 100m
dataset = dataset.transform(to_dimensions=(100, 100))

print(dataset.pitch_dimensions)  # (100.0, 100.0)
```

!!! tip "Zone-Based Scaling"
    Dimension transforms use **zone-based scaling** that preserves IFAB standard pitch feature proportions. The penalty area, six-yard box, center circle, and other markings remain correctly proportioned relative to the pitch, rather than being uniformly stretched.

## Chaining Transforms

Transforms can be chained by calling `.transform()` multiple times:

```python
result = (
    dataset
    .transform(to_orientation="home_away")
    .transform(to_dimensions=(100, 100))
    .transform(to_coordinates="opta")
)
```

Or applied all at once:

```python
result = dataset.transform(
    to_orientation="home_away",
    to_dimensions=(100, 100),
    to_coordinates="opta",
)
```

Both approaches produce the same result.

## Metadata Updates

After a transform, the metadata DataFrame is updated to reflect the new state:

```python
dataset = dataset.transform(to_coordinates="opta")

# Metadata reflects the new coordinate system
print(dataset.metadata["coordinate_system"][0])  # "opta"
print(dataset.coordinate_system)                  # "opta"
```

The `coordinate_system`, `orientation`, `pitch_length`, and `pitch_width` fields are all updated automatically.
