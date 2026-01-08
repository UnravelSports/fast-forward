use pyo3::prelude::*;
use pyo3::types::PyModule;

mod coordinates;
mod dataframe;
mod error;
mod models;
mod orientation;
mod providers;

/// kloppy-light: Fast tracking data loader
#[pymodule]
fn _kloppy_light(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register secondspectrum submodule
    let secondspectrum = PyModule::new_bound(m.py(), "secondspectrum")?;
    providers::secondspectrum::register_module(&secondspectrum)?;
    m.add_submodule(&secondspectrum)?;

    // Register skillcorner submodule
    let skillcorner = PyModule::new_bound(m.py(), "skillcorner")?;
    providers::skillcorner::register_module(&skillcorner)?;
    m.add_submodule(&skillcorner)?;

    Ok(())
}
