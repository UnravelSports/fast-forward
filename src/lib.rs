use pyo3::prelude::*;
use pyo3::types::PyModule;

mod coordinates;
mod dataframe;
mod error;
mod filter_pushdown;
mod models;
mod orientation;
mod providers;
mod transforms;

/// kloppy-light: Fast tracking data loader
#[pymodule]
fn _kloppy_light(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register cdf submodule
    let cdf = PyModule::new(m.py(), "cdf")?;
    providers::cdf::register_module(&cdf)?;
    m.add_submodule(&cdf)?;

    // Register gradientsports submodule
    let gradientsports = PyModule::new(m.py(), "gradientsports")?;
    providers::gradientsports::register_module(&gradientsports)?;
    m.add_submodule(&gradientsports)?;

    // Register secondspectrum submodule
    let secondspectrum = PyModule::new(m.py(), "secondspectrum")?;
    providers::secondspectrum::register_module(&secondspectrum)?;
    m.add_submodule(&secondspectrum)?;

    // Register skillcorner submodule
    let skillcorner = PyModule::new(m.py(), "skillcorner")?;
    providers::skillcorner::register_module(&skillcorner)?;
    m.add_submodule(&skillcorner)?;

    // Register sportec submodule
    let sportec = PyModule::new(m.py(), "sportec")?;
    providers::sportec::register_module(&sportec)?;
    m.add_submodule(&sportec)?;

    // Register hawkeye submodule
    let hawkeye = PyModule::new(m.py(), "hawkeye")?;
    providers::hawkeye::register_module(&hawkeye)?;
    m.add_submodule(&hawkeye)?;

    // Register tracab submodule
    let tracab = PyModule::new(m.py(), "tracab")?;
    providers::tracab::register_module(&tracab)?;
    m.add_submodule(&tracab)?;

    // Register transforms submodule
    let transforms_mod = PyModule::new(m.py(), "transforms")?;
    transforms::register_module(&transforms_mod)?;
    m.add_submodule(&transforms_mod)?;

    Ok(())
}
