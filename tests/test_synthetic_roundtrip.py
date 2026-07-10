"""Round-trip guard for the synthetic provider-data generator.

``scripts/synthesize_provider_data.py`` re-encodes a real CDF match into the
native on-disk format of the 5 "gap" providers (secondspectrum, gradientsports,
signality, statsperform, scisports) so they can be benchmarked. This test asserts
that each writer produces files that load *back* through the provider's own
``load_tracking()`` with the correct structure, a center-origin coordinate cloud
(catches a missed inverse transform / the scisports x/y swap) and a frame count
that matches the provider's native FPS (the whole point of the re-encoding).

It runs on the tiny committed CDF fixture (``tests/files/cdf_*``), so it needs no
large data, no GPU and no network — safe for CI.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import synthesize_provider_data as syn  # noqa: E402
from tests import config as C  # noqa: E402


@pytest.fixture(scope="module")
def canonical():
    """The tiny CDF fixture loaded into the provider-agnostic representation."""
    return syn.load_canonical(C.CDF_RAW, C.CDF_META)


def test_canonical_has_two_periods_and_full_squads(canonical):
    assert canonical.periods == [1, 2]
    assert len(canonical.players) >= 22
    assert all(len(canonical.frames[p]) > 0 for p in canonical.periods)


@pytest.mark.parametrize("provider", list(syn.WRITERS))
def test_writer_roundtrips_through_loader(provider, canonical, tmp_path):
    result = syn.WRITERS[provider](canonical, tmp_path, minutes=None)

    # Files were actually written and are non-empty.
    for fp in result["files"]:
        assert fp.exists() and fp.stat().st_size > 0, f"{provider}: missing/empty {fp}"

    # verify() loads the file(s) back through the provider loader and asserts:
    # 5 non-None frames, teams == 2, players >= 22, both periods present,
    # center-origin coordinate cloud (x/y medians near 0, x-spread > y-spread —
    # this catches a missed inverse transform or the scisports x/y swap), and a
    # distinct-frame count within 1% of native_fps * period_seconds.
    expected = sum(
        syn.target_count(canonical, p, syn.PROVIDER_FPS[provider], None)
        for p in canonical.periods
    )
    hint = syn.verify(result, expected_frames=expected)
    assert hint["rows"] > 0
