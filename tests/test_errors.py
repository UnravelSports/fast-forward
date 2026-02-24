"""Tests for error enrichment with GitHub issue URL.

Verifies that all error messages include the GitHub issues URL footer
and that format errors include the format hint.
"""

import pytest

from fastforward._errors import (
    GITHUB_ISSUES_URL,
    _is_format_error,
    _has_github_url,
    _enrich_exception,
    error_handler,
    with_error_handler,
)
from fastforward import (
    secondspectrum,
    skillcorner,
    sportec,
    tracab,
    cdf,
    hawkeye,
    signality,
    respovision,
    statsperform,
    gradientsports,
)


# =============================================================================
# Unit tests for _errors.py helpers
# =============================================================================


class TestIsFormatError:
    """Tests for _is_format_error() detection."""

    def test_unknown_metadata_format(self):
        assert _is_format_error("Invalid input: Unknown metadata format: expected JSON or XML")

    def test_json_parsing_error(self):
        assert _is_format_error("JSON parsing error: expected value at line 1")

    def test_xml_parsing_error(self):
        assert _is_format_error("XML parsing error: unexpected end of input")

    def test_xml_attribute_parsing_error(self):
        assert _is_format_error("XML attribute parsing error: duplicate attribute")

    def test_expected_json_or_xml(self):
        assert _is_format_error("Expected JSON or XML content")

    def test_case_insensitive(self):
        assert _is_format_error("JSON PARSING ERROR: something")
        assert _is_format_error("xml Parsing Error: something")

    def test_regular_error_not_format(self):
        assert not _is_format_error("IO error: file not found")

    def test_polars_error_not_format(self):
        assert not _is_format_error("Polars error: column not found")

    def test_empty_string(self):
        assert not _is_format_error("")


class TestHasGithubUrl:
    """Tests for _has_github_url() duplicate detection."""

    def test_detects_url_without_www(self):
        msg = "Error at https://github.com/UnravelSports/fast-forward/issues"
        assert _has_github_url(msg)

    def test_detects_url_with_www(self):
        msg = "Error at https://www.github.com/UnravelSports/fast-forward/issues"
        assert _has_github_url(msg)

    def test_detects_url_in_longer_message(self):
        msg = (
            "Schema mismatch: missing field. Please report at "
            "https://github.com/UnravelSports/fast-forward/issues with details."
        )
        assert _has_github_url(msg)

    def test_no_url(self):
        assert not _has_github_url("IO error: file not found")

    def test_different_repo_url(self):
        assert not _has_github_url("See https://github.com/other/repo/issues")

    def test_empty_string(self):
        assert not _has_github_url("")


class TestEnrichException:
    """Tests for _enrich_exception() modification."""

    def test_appends_footer(self):
        e = ValueError("something went wrong")
        _enrich_exception(e)
        assert "something went wrong" in str(e)
        assert GITHUB_ISSUES_URL in str(e)

    def test_preserves_exception_type(self):
        e = TypeError("bad type")
        _enrich_exception(e)
        assert isinstance(e, TypeError)

    def test_adds_format_hint_for_format_errors(self):
        e = ValueError("JSON parsing error: unexpected token")
        _enrich_exception(e)
        msg = str(e)
        assert "might not be supported yet" in msg
        assert GITHUB_ISSUES_URL in msg

    def test_no_format_hint_for_regular_errors(self):
        e = ValueError("IO error: file not found")
        _enrich_exception(e)
        msg = str(e)
        assert "might not be supported" not in msg
        assert GITHUB_ISSUES_URL in msg

    def test_skips_if_url_already_present(self):
        original_msg = f"Error. Report at {GITHUB_ISSUES_URL}"
        e = ValueError(original_msg)
        _enrich_exception(e)
        assert str(e) == original_msg

    def test_no_double_footer(self):
        e = ValueError("something went wrong")
        _enrich_exception(e)
        _enrich_exception(e)
        msg = str(e)
        assert msg.count(GITHUB_ISSUES_URL) == 1


class TestErrorHandler:
    """Tests for the error_handler context manager."""

    def test_enriches_on_exception(self):
        with pytest.raises(ValueError, match=GITHUB_ISSUES_URL):
            with error_handler():
                raise ValueError("test error")

    def test_preserves_original_message(self):
        with pytest.raises(ValueError, match="test error"):
            with error_handler():
                raise ValueError("test error")

    def test_preserves_exception_type(self):
        with pytest.raises(TypeError):
            with error_handler():
                raise TypeError("type error")

    def test_no_exception_passes_through(self):
        with error_handler():
            pass  # no error

    def test_format_hint_added(self):
        with pytest.raises(ValueError, match="might not be supported"):
            with error_handler():
                raise ValueError("JSON parsing error: bad token")


class TestWithErrorHandlerDecorator:
    """Tests for the with_error_handler decorator."""

    def test_enriches_on_exception(self):
        @with_error_handler
        def failing():
            raise ValueError("decorated error")

        with pytest.raises(ValueError, match=GITHUB_ISSUES_URL):
            failing()

    def test_preserves_return_value(self):
        @with_error_handler
        def success():
            return 42

        assert success() == 42

    def test_preserves_function_name(self):
        @with_error_handler
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


# =============================================================================
# Integration tests: GitHub URL in provider error messages
# =============================================================================

GITHUB_URL_PATTERN = "github.com/UnravelSports/fast-forward/issues"


class TestGitHubUrlInErrors:
    """Verify every provider includes GitHub URL in error messages."""

    def test_secondspectrum_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            secondspectrum.load_tracking(b"invalid", b"invalid")

    def test_skillcorner_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            skillcorner.load_tracking(b"invalid", b"invalid")

    def test_sportec_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            sportec.load_tracking(b"invalid", b"invalid")

    def test_tracab_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            tracab.load_tracking(b"invalid", b"invalid")

    def test_cdf_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            cdf.load_tracking(b"invalid", b"invalid")

    def test_hawkeye_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            hawkeye.load_tracking([b"invalid"], [b"invalid"], b"invalid")

    def test_signality_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            signality.load_tracking(b"invalid", [b"invalid"], b"invalid")

    def test_respovision_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            respovision.load_tracking(b"invalid")

    def test_statsperform_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            statsperform.load_tracking(b"invalid", b"invalid")

    def test_gradientsports_error_has_github_url(self):
        with pytest.raises(Exception, match=GITHUB_URL_PATTERN):
            gradientsports.load_tracking(b"invalid", b"invalid", b"invalid")


class TestFormatHintInErrors:
    """Verify format errors include the 'might not be supported' hint."""

    def test_format_hint_in_enriched_error(self):
        """Test that a format-related error gets the hint."""
        with pytest.raises(
            ValueError,
            match="might not be supported",
        ):
            with error_handler():
                raise ValueError(
                    "Invalid input: Unknown metadata format: expected JSON or XML"
                )
