from astropy.io import fits
from astropy.wcs import WCS
from sip_tpv.sip_to_pv import sip_to_pv
from sip_tpv.pv_to_sip import pv_to_sip


def test_sip2pv():
    sipheader = fits.Header.fromtextfile('data/IRAC_3.6um_sip.txt')
    myheader = sipheader.copy()
    sip_to_pv(myheader)

    wsip = WCS(sipheader)
    wtpv = WCS(myheader)

    world1 = wsip.all_pix2world([[1, 1]], 1)
    world2 = wtpv.all_pix2world([[1, 1]], 1)

    # map will report None for a value in the case that one array is longer than the other, raising an error
    for i, j in map(None, world1, world2):
        for q, k in map(None, i, j):
            assert q == k


def test_pv2sip():
    pvheader = fits.Header.fromtextfile('data/PTF_r_chip01_tpv.txt')
    header = pvheader.copy()
    pv_to_sip(header)

    wsip = WCS(pvheader)
    wtpv = WCS(header)

    world1 = wsip.all_pix2world([[1, 1]], 1)
    world2 = wtpv.all_pix2world([[1, 1]], 1)

    for i, j in map(None, world1, world2):
        for q, k in map(None, i, j):
            assert q == k

    sip_to_pv(header)
    wtpv = WCS(header)
    world1 = wsip.all_world2pix([[1, 1]], 1)
    world2 = wtpv.all_world2pix([[1, 1]], 1)

    for i, j in map(None, world2, world1):
        for q, k in map(None, i, j):
            print q, k
            """
            Prints out:
            -25.8095954324 -25.8095954322
            19.1907457483 19.1907457481
            """
            assert q == k
