use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KloppyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Missing metadata: {0}")]
    MissingMetadata(String),

    #[error("Unsupported coordinate system: {0}")]
    UnsupportedCoordinates(String),

    #[error("Unsupported layout: {0}")]
    UnsupportedLayout(String),
}

impl From<KloppyError> for PyErr {
    fn from(err: KloppyError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
