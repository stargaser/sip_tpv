from pathlib import Path

import numpy as np
import numpy.testing as npt
from astropy.io import fits
from astropy.wcs import WCS

from sip_tpv import pv_to_sip, sip_to_pv

DATA_DIR = Path(__file__).parent / "data"


def test_sip2pv():
    """Ensure SIP->PV conversion preserves world/pixel transforms."""
    sip_header = fits.Header.fromtextfile(DATA_DIR / "IRAC_3.6um_sip.txt")
    control_header = sip_header.copy()
    naxis1 = sip_header['NAXIS1']
    naxis2 = sip_header['NAXIS2']
    x = np.linspace(1, naxis1, 10)
    y = np.linspace(1, naxis2, 10)
    xx, yy = np.meshgrid(x, y)
    pixargs = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T

    sip_to_pv(sip_header)

    wsip = WCS(sip_header)
    wtpv = WCS(control_header)

    world1 = wsip.all_pix2world(pixargs, 1)
    world2 = wtpv.all_pix2world(pixargs, 1)

    npt.assert_allclose(world1, world2, rtol=1e-12)

    pix1 = wsip.all_world2pix(world1, 1)
    pix2 = wtpv.all_world2pix(world2, 1)
    npt.assert_almost_equal(pix1, pixargs, 4)
    npt.assert_almost_equal(pix2, pixargs, 4)


def test_pv2sip():
    """Ensure PV->SIP conversion round-trips world/pixel transformations."""
    pv_header = fits.Header.fromtextfile(DATA_DIR / "PTF_r_chip01_tpv.txt")
    control_header = pv_header.copy()
    naxis1 = pv_header['NAXIS1']
    naxis2 = pv_header['NAXIS2']
    x = np.linspace(1, naxis1, 10)
    y = np.linspace(1, naxis2, 10)
    xx, yy = np.meshgrid(x, y)
    pixargs = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T

    pv_to_sip(pv_header)

    wsip = WCS(pv_header)
    wtpv = WCS(control_header)

    world1 = wsip.all_pix2world(pixargs, 1)
    world2 = wtpv.all_pix2world(pixargs, 1)

    npt.assert_allclose(world1, world2, rtol=1e-12)

    pix1 = wsip.all_world2pix(world1, 1)
    pix2 = wtpv.all_world2pix(world2, 1)

    npt.assert_almost_equal(pix1, pixargs, 4)
    npt.assert_almost_equal(pix2, pixargs, 4)


def test_sip2pv_without_reverse_coefficients():
    """SIP->PV works when AP_ORDER/BP_ORDER are absent (issue #1)."""
    sip_header = fits.Header.fromtextfile(DATA_DIR / "IRAC_3.6um_sip.txt")
    control_header = sip_header.copy()
    for key in [k for k in sip_header if k.startswith(("AP_", "BP_"))]:
        del sip_header[key]
    assert "AP_ORDER" not in sip_header

    sip_to_pv(sip_header)
    sip_to_pv(control_header)

    pv_keys = [k for k in control_header if k.startswith("PV")]
    assert pv_keys
    for key in pv_keys:
        assert sip_header[key] == control_header[key], key
