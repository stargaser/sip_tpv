"""Write tests/data/reference_coefficients.json for test_reference.py.

The reference was generated with the sip_tpv 1.2 release from PyPI, so the
tests compare the current code against an independent, known-good result:

    uv run --isolated --no-project --with sip_tpv==1.2 python tests/make_reference.py
"""

import json
import re
from importlib.metadata import version
from pathlib import Path

import astropy
import numpy as np
from astropy.io import fits

from sip_tpv import pv_to_sip, sip_to_pv

DATA_DIR = Path(__file__).parent / "data"
DISTORTION_KEY = re.compile(r"^(A|B|AP|BP)_(\d+_\d+|ORDER)$|^PV\d_\d+$|^CTYPE\d$")


def distortion_keywords(header):
    return {k: header[k] for k in header if DISTORTION_KEY.match(k)}


def compute():
    sip_header = fits.Header.fromtextfile(DATA_DIR / "IRAC_3.6um_sip.txt")
    sip_to_pv(sip_header)
    tpv_header = fits.Header.fromtextfile(DATA_DIR / "PTF_r_chip01_tpv.txt")
    pv_to_sip(tpv_header)
    return {"sip_to_pv/IRAC_3.6um_sip": distortion_keywords(sip_header),
            "pv_to_sip/PTF_r_chip01_tpv": distortion_keywords(tpv_header)}


if __name__ == "__main__":
    reference = {
        "generated_with": {"sip_tpv": version("sip_tpv"),
                           "numpy": np.__version__,
                           "astropy": astropy.__version__},
        "cases": compute(),
    }
    path = DATA_DIR / "reference_coefficients.json"
    path.write_text(json.dumps(reference, indent=1) + "\n")
    print(f"wrote {path}")
