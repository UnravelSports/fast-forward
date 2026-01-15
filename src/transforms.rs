//! Post-load coordinate, orientation, and dimension transformation functions.
//!
//! These functions are exposed to Python to enable post-load transformation of tracking data.
//! They work on Polars DataFrames and apply transformations to x, y, z coordinate columns.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::coordinates::{
    transform_from_cdf, transform_to_cdf, CoordinateSystem, DimensionTransform,
};

// ============================================================================
// Coordinate system transformation
// ============================================================================

/// Transform DataFrame coordinates between coordinate systems.
///
/// Uses CDF as intermediate format: source → CDF → target
///
/// # Arguments
/// * `df` - DataFrame with x, y, z columns
/// * `from_system` - Source coordinate system name
/// * `to_system` - Target coordinate system name
/// * `pitch_length` - Pitch length in meters
/// * `pitch_width` - Pitch width in meters
#[pyfunction]
#[pyo3(signature = (df, from_system, to_system, pitch_length, pitch_width))]
pub fn transform_coordinates(
    df: PyDataFrame,
    from_system: &str,
    to_system: &str,
    pitch_length: f32,
    pitch_width: f32,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = df.into();

    let from_cs = CoordinateSystem::from_str(from_system)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let to_cs = CoordinateSystem::from_str(to_system)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // If same system, return unchanged
    if from_cs == to_cs {
        return Ok(PyDataFrame(df));
    }

    // Check for required columns
    let has_x = df.column("x").is_ok();
    let has_y = df.column("y").is_ok();
    let has_z = df.column("z").is_ok();

    if !has_x || !has_y {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DataFrame must have 'x' and 'y' columns",
        ));
    }

    // Transform coordinates
    let x_col = df.column("x").unwrap();
    let y_col = df.column("y").unwrap();

    let x_vals: Vec<Option<f32>> = x_col.f32().unwrap().into_iter().collect();
    let y_vals: Vec<Option<f32>> = y_col.f32().unwrap().into_iter().collect();

    let z_vals: Vec<Option<f32>> = if has_z {
        df.column("z").unwrap().f32().unwrap().into_iter().collect()
    } else {
        vec![Some(0.0); x_vals.len()]
    };

    let mut new_x: Vec<Option<f32>> = Vec::with_capacity(x_vals.len());
    let mut new_y: Vec<Option<f32>> = Vec::with_capacity(y_vals.len());
    let mut new_z: Vec<Option<f32>> = Vec::with_capacity(z_vals.len());

    for i in 0..x_vals.len() {
        match (x_vals[i], y_vals[i], z_vals[i]) {
            (Some(x), Some(y), Some(z)) => {
                // Transform via CDF
                let (cdf_x, cdf_y, cdf_z) =
                    transform_to_cdf(x, y, z, from_cs, pitch_length, pitch_width);
                let (tx, ty, tz) =
                    transform_from_cdf(cdf_x, cdf_y, cdf_z, to_cs, pitch_length, pitch_width);
                new_x.push(Some(tx));
                new_y.push(Some(ty));
                new_z.push(Some(tz));
            }
            _ => {
                new_x.push(None);
                new_y.push(None);
                new_z.push(None);
            }
        }
    }

    // Build new DataFrame with transformed columns
    let mut columns: Vec<Column> = Vec::new();

    for col in df.get_columns() {
        let name = col.name();
        if name == "x" {
            columns.push(
                Series::new(PlSmallStr::from_static("x"), &new_x).into_column()
            );
        } else if name == "y" {
            columns.push(
                Series::new(PlSmallStr::from_static("y"), &new_y).into_column()
            );
        } else if name == "z" && has_z {
            columns.push(
                Series::new(PlSmallStr::from_static("z"), &new_z).into_column()
            );
        } else {
            columns.push(col.clone());
        }
    }

    let result = DataFrame::new(columns)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyDataFrame(result))
}

// ============================================================================
// Dimension transformation (zone-based)
// ============================================================================

/// Transform DataFrame to different pitch dimensions using zone-based scaling.
///
/// Uses IFAB standard zone boundaries to preserve pitch feature proportions
/// (penalty area, six-yard box, center circle, etc.).
///
/// # Arguments
/// * `df` - DataFrame with x, y columns (must be in CDF format: center origin, meters)
/// * `from_length` - Source pitch length in meters
/// * `from_width` - Source pitch width in meters
/// * `to_length` - Target pitch length in meters
/// * `to_width` - Target pitch width in meters
#[pyfunction]
#[pyo3(signature = (df, from_length, from_width, to_length, to_width))]
pub fn transform_dimensions(
    df: PyDataFrame,
    from_length: f32,
    from_width: f32,
    to_length: f32,
    to_width: f32,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = df.into();

    let transform = DimensionTransform::new(from_length, from_width, to_length, to_width);

    // If identity transform, return unchanged
    if transform.is_identity() {
        return Ok(PyDataFrame(df));
    }

    // Check for required columns
    let has_x = df.column("x").is_ok();
    let has_y = df.column("y").is_ok();

    if !has_x || !has_y {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DataFrame must have 'x' and 'y' columns",
        ));
    }

    // Transform coordinates
    let x_col = df.column("x").unwrap();
    let y_col = df.column("y").unwrap();

    let x_vals: Vec<Option<f32>> = x_col.f32().unwrap().into_iter().collect();
    let y_vals: Vec<Option<f32>> = y_col.f32().unwrap().into_iter().collect();

    let new_x: Vec<Option<f32>> = x_vals
        .iter()
        .map(|x| x.map(|v| transform.transform_x(v)))
        .collect();

    let new_y: Vec<Option<f32>> = y_vals
        .iter()
        .map(|y| y.map(|v| transform.transform_y(v)))
        .collect();

    // Build new DataFrame with transformed columns
    let mut columns: Vec<Column> = Vec::new();

    for col in df.get_columns() {
        let name = col.name();
        if name == "x" {
            columns.push(
                Series::new(PlSmallStr::from_static("x"), &new_x).into_column()
            );
        } else if name == "y" {
            columns.push(
                Series::new(PlSmallStr::from_static("y"), &new_y).into_column()
            );
        } else {
            columns.push(col.clone());
        }
    }

    let result = DataFrame::new(columns)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyDataFrame(result))
}

// ============================================================================
// Orientation transformation
// ============================================================================

/// Transform DataFrame orientation by flipping coordinates.
///
/// Orientation flipping negates x and y coordinates around the center (0, 0).
/// This is used to ensure consistent attacking direction.
///
/// # Arguments
/// * `df` - DataFrame with x, y columns (must be in CDF format: center origin)
/// * `flip` - If true, flip the coordinates (negate x and y)
#[pyfunction]
#[pyo3(signature = (df, flip))]
pub fn transform_orientation(
    df: PyDataFrame,
    flip: bool,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = df.into();

    // If no flip needed, return unchanged
    if !flip {
        return Ok(PyDataFrame(df));
    }

    // Check for required columns
    let has_x = df.column("x").is_ok();
    let has_y = df.column("y").is_ok();

    if !has_x || !has_y {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DataFrame must have 'x' and 'y' columns",
        ));
    }

    // Flip coordinates (negate x and y)
    let x_col = df.column("x").unwrap();
    let y_col = df.column("y").unwrap();

    let x_vals: Vec<Option<f32>> = x_col.f32().unwrap().into_iter().collect();
    let y_vals: Vec<Option<f32>> = y_col.f32().unwrap().into_iter().collect();

    let new_x: Vec<Option<f32>> = x_vals.iter().map(|x| x.map(|v| -v)).collect();
    let new_y: Vec<Option<f32>> = y_vals.iter().map(|y| y.map(|v| -v)).collect();

    // Build new DataFrame with flipped columns
    let mut columns: Vec<Column> = Vec::new();

    for col in df.get_columns() {
        let name = col.name();
        if name == "x" {
            columns.push(
                Series::new(PlSmallStr::from_static("x"), &new_x).into_column()
            );
        } else if name == "y" {
            columns.push(
                Series::new(PlSmallStr::from_static("y"), &new_y).into_column()
            );
        } else {
            columns.push(col.clone());
        }
    }

    let result = DataFrame::new(columns)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyDataFrame(result))
}

// ============================================================================
// Module registration
// ============================================================================

/// Register transforms submodule
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(transform_dimensions, m)?)?;
    m.add_function(wrap_pyfunction!(transform_orientation, m)?)?;
    Ok(())
}
