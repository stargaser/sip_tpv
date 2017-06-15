from __future__ import print_function
import os
import numpy.testing as npt
from astropy.io import fits
from astropy.wcs import WCS
from ..sip_to_pv import sip_to_pv
from ..pv_to_sip import pv_to_sip

dir_name = os.path.split(__file__)[0]


def test_sip2pv():
    sip_header = fits.Header.fromtextfile(os.path.join(dir_name, 'data/IRAC_3.6um_sip.txt'))
    control_header = sip_header.copy()
    sip_to_pv(sip_header)

    wsip = WCS(sip_header)
    wtpv = WCS(control_header)

    world1 = wsip.all_pix2world([[1, 1]], 1)
    world2 = wtpv.all_pix2world([[1, 1]], 1)
    print("Test1")
    npt.assert_array_almost_equal(world1, world2, 6)


def test_():
    pv_header = fits.Header.fromtextfile(os.path.join(dir_name, 'data/IRAC_3.6um_sip.txt'))
    control_header = pv_header.copy()
    pv_to_sip(pv_header)

    wsip = WCS(pv_header)
    wtpv = WCS(control_header)

    world1 = wsip.all_pix2world([[1, 1]], 1)
    world2 = wtpv.all_pix2world([[1, 1]], 1)
    print("Test2")
    npt.assert_array_almost_equal(world1, world2, 3)

    sip_to_pv(pv_header)

    world1 = wsip.all_pix2world([[1, 1]], 1)
    print("Test3")
    npt.assert_almost_equal(world1, world2, 3)
