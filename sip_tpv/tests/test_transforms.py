from __future__ import print_function
import os
import numpy.testing as npt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ..sip_to_pv import sip_to_pv
from ..pv_to_sip import pv_to_sip

dir_name = os.path.split(__file__)[0]


def test_sip2pv():
    """
    Test conversion of sip 2 pv keywords, ensure that both provide same ra/dec <--> x/y transforms.
    """

    sip_header = fits.Header.fromtextfile(os.path.join(dir_name, 'data/IRAC_3.6um_sip.txt'))
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

    npt.assert_equal(world1, world2)

    pix1 = wsip.all_world2pix(world1, 1)
    pix2 = wtpv.all_world2pix(world2, 1)
    npt.assert_almost_equal(pix1, pixargs, 4)
    npt.assert_almost_equal(pix2, pixargs, 4)


def test_pv2sip():
    """
    Test conversion of pv 2 sip keywords, check to see that world2pix transform round trips and is equal for pv and
     sip keywords.
    """

    pv_header = fits.Header.fromtextfile(os.path.join(dir_name, 'data/PTF_r_chip01_tpv.txt'))
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

    npt.assert_equal(world1, world2)

    pix1 = wsip.all_world2pix(world1, 1)
    pix2 = wtpv.all_world2pix(world2, 1)

    npt.assert_almost_equal(pix1, pixargs, 4)
    npt.assert_almost_equal(pix2, pixargs, 4)
