"""Error enrichment utilities.

Appends GitHub issue URL and format hints to error messages
so users always see actionable reporting information.
"""

import contextlib
import functools

GITHUB_ISSUES_URL = "https://github.com/UnravelSports/fast-forward/issues"

_GITHUB_FOOTER = (
    "\n\n"
    "If this error persists, please report it at "
    f"{GITHUB_ISSUES_URL} with:\n"
    "  - This error message\n"
    "  - A minimal reproducible example\n"
    "  - An anonymized data sample (first 5-10 lines)"
)

_FORMAT_HINT = (
    "\n\nHint: The data format you are using might not be supported yet."
)

# Patterns that indicate a format/parsing error (checked case-insensitively)
_FORMAT_ERROR_PATTERNS = [
    "unknown metadata format",
    "expected json or xml",
    "json parsing error",
    "xml parsing error",
    "xml attribute parsing error",
]


def _is_format_error(message: str) -> bool:
    """Check if an error message indicates a data format problem."""
    msg_lower = message.lower()
    return any(pattern in msg_lower for pattern in _FORMAT_ERROR_PATTERNS)


def _has_github_url(message: str) -> bool:
    """Check if message already contains the GitHub issues URL.

    Handles both www. and non-www. variants to prevent duplicates.
    """
    return "github.com/UnravelSports/fast-forward/issues" in message


def _enrich_exception(e: Exception) -> None:
    """Add GitHub footer (and format hint if applicable) to an exception's message."""
    msg = str(e)
    # Don't add footer if already present
    if _has_github_url(msg):
        return

    # Add format hint if applicable
    hint = _FORMAT_HINT if _is_format_error(msg) else ""

    # Modify the error args to include footer
    new_msg = f"{msg}{hint}{_GITHUB_FOOTER}"
    e.args = (new_msg,) + e.args[1:]


@contextlib.contextmanager
def error_handler():
    """Context manager that enriches exceptions with GitHub issue URL.

    Catches any Exception, appends a format hint (if applicable) and
    the GitHub footer to the error message, then re-raises with the
    original exception type preserved.

    Skips enrichment if the URL is already present (e.g., Rust SchemaMismatch).
    """
    try:
        yield
    except Exception as e:
        _enrich_exception(e)
        raise


def with_error_handler(func):
    """Decorator that wraps a function in error_handler context.

    Use on public API functions (load_tracking, load_metadata_only, etc.)
    to ensure all errors include the GitHub issue URL footer.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with error_handler():
            return func(*args, **kwargs)
    return wrapper
