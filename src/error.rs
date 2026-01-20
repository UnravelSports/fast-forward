use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

/// URL for submitting bug reports with schema mismatches.
/// TODO: Update this URL before launch!
pub const GITHUB_ISSUES_URL: &str = "https://www.github.com/UnravelSports/kloppy-light/issues";

#[derive(Error, Debug)]
pub enum KloppyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("XML parsing error: {0}")]
    Xml(#[from] quick_xml::Error),

    #[error("XML attribute parsing error: {0}")]
    XmlAttr(#[from] quick_xml::events::attributes::AttrError),

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

    #[error("Empty data: {context} data is empty or contains only whitespace")]
    EmptyData { context: String },

    #[error("Corrupted data at line {line}: {message}")]
    CorruptedData { message: String, line: usize },

    #[error(
        "Schema mismatch: {message}. This may indicate a data format change. \
         Please submit a GitHub issue at {url} with:\n\
         - This error message\n\
         - A minimal reproducible example\n\
         - An anonymized data sample (first 5-10 lines)",
        url = GITHUB_ISSUES_URL
    )]
    SchemaMismatch { message: String },
}

impl From<KloppyError> for PyErr {
    fn from(err: KloppyError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Validate that data is not empty or whitespace-only.
///
/// # Arguments
/// * `data` - The byte slice to validate
/// * `context` - A description of what the data represents (e.g., "tracking", "metadata")
///
/// # Returns
/// * `Ok(())` if the data is non-empty and contains non-whitespace characters
/// * `Err(KloppyError::EmptyData)` if the data is empty or whitespace-only
pub fn validate_not_empty(data: &[u8], context: &str) -> Result<(), KloppyError> {
    if data.is_empty() || data.iter().all(|b| b.is_ascii_whitespace()) {
        return Err(KloppyError::EmptyData {
            context: context.to_string(),
        });
    }
    Ok(())
}

/// Categorize a JSON parsing error into the appropriate KloppyError variant.
///
/// This helps distinguish between:
/// - Syntax errors (corrupted data) - recoverable if you can skip the line
/// - Schema mismatches (missing/wrong fields) - likely a data format change
///
/// # Arguments
/// * `e` - The serde_json error
/// * `line_num` - The line number where the error occurred (1-indexed)
/// * `sample` - A sample of the data that failed to parse (for context in error message)
///
/// # Returns
/// * `KloppyError::CorruptedData` for syntax errors
/// * `KloppyError::SchemaMismatch` for missing field or type errors
pub fn categorize_json_error(e: serde_json::Error, line_num: usize, sample: &str) -> KloppyError {
    let sample_preview = if sample.len() > 50 {
        format!("{}...", &sample[..50])
    } else {
        sample.to_string()
    };

    if e.is_syntax() || e.is_eof() {
        KloppyError::CorruptedData {
            message: format!("{} (sample: {})", e, sample_preview),
            line: line_num,
        }
    } else {
        // Missing field, invalid type, etc. - likely a schema change
        KloppyError::SchemaMismatch {
            message: format!("{} at line {} (sample: {})", e, line_num, sample_preview),
        }
    }
}
