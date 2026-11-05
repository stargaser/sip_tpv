"""Compare converted coefficients against output from the sip_tpv 1.2 release."""

import json
import re
from pathlib import Path

import numpy.testing as npt
import pytest

from make_reference import compute

DATA_DIR = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA_DIR / "reference_coefficients.json").read_text())
SIP_COEFF = re.compile(r"^(A|B|AP|BP)_(\d+)_(\d+)$")

# SIP terms are compared by their size in pixels this far from CRPIX, so
# that high-order terms (~1e-20) are checked as strictly as low-order ones.
SIP_RADIUS = 1000.0


@pytest.fixture(scope="module")
def current():
    return compute()


@pytest.mark.parametrize("case", sorted(REFERENCE["cases"]))
def test_matches_reference(current, case):
    expected = REFERENCE["cases"][case]
    actual = current[case]
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, str) or key.endswith("_ORDER"):
            assert actual[key] == value, key
        elif match := SIP_COEFF.match(key):
            scale = SIP_RADIUS ** (int(match[2]) + int(match[3]))
            npt.assert_allclose(actual[key] * scale, value * scale,
                                rtol=1e-9, atol=1e-9, err_msg=key)
        else:
            npt.assert_allclose(actual[key], value,
                                rtol=1e-9, atol=1e-12, err_msg=key)
