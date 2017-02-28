#!/usr/bin/env python

from __future__ import print_function, absolute_import, division

# Licensed under a 3-clause BSD style license - see LICENSE.txt
"""
Convert SIP convention distortion keywords to the TPV convention.

This module includes the equations for converting from the SIP distortion representation
 to the PV or TPV representations, following the SPIE proceedings paper at
     http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1363103
     and http://web.ipac.caltech.edu/staff/shupe/reprints/SIP_to_PV_SPIE2012.pdf .
The work described in that paper is extended to 7th order.

Copyright (c) 2012-2014, California Institute of Technology

If you make use of this work, please cite:
"More flexibility in representing geometric distortion in astronomical images,"
  Shupe, David L.; Laher, Russ R.; Storrie-Lombardi, Lisa; Surace, Jason; Grillmair, Carl;
  Levitan, David; Sesar, Branimir, 2012, in Software and Cyberinfrastructure for
  Astronomy II. Proceedings of the SPIE, Volume 8451, article id. 84511M.

Thanks to Octavi Fors for contributing code modifications for better modularization,
   and for extensive testing.

Funding is acknowledged from NASA to the NASA Herschel Science Center and the
   Spitzer Science Center.

Contact: David Shupe, IPAC/Caltech.

"""

version = 1.0

from sympy import symbols, Matrix, collect, simplify, poly
from sympy.tensor import IndexedBase, Idx
import numpy as np
import astropy.io.fits as pyfits
from collections import OrderedDict
import os

def calc_tpvexprs():
    """ calculate the Sympy expression for TPV distortion

    Parameters:
    -----------
    None

    Returns:
    --------
    pvrange (list) : indices to the PV keywords
    tpvx (Sympy expr) : equation for x-distortion in TPV convention
    tpvy (Sympy expr) : equation for y-distortion in TPV convention
    """
    x, y = symbols("x y")
    pvrange = list(range(0,39))
    pvrange.remove(3)
    pvrange.remove(11)
    pvrange.remove(23)
    pv1 = symbols('pv1_0:39')
    pv2 = symbols('pv2_0:39')

    # Copy the equations from the PV-to-SIP paper and convert to code,
    #  leaving out radial terms PV[1,3], PV[1,11], PV[1,23], PV[1,39]

    tpvx = pv1[0] + pv1[1]*x + pv1[2]*y + pv1[4]*x**2 + pv1[5]*x*y + pv1[6]*y**2 + \
        pv1[7]*x**3 + pv1[8]*x**2*y + pv1[9]*x*y**2 + pv1[10]*y**3 +  \
        pv1[12]*x**4 + pv1[13]*x**3*y + pv1[14]*x**2*y**2 + pv1[15]*x*y**3 + pv1[16]*y**4 + \
        pv1[17]*x**5 + pv1[18]*x**4*y + pv1[19]*x**3*y**2 + pv1[20]*x**2*y**3 + pv1[21]*x*y**4 + \
        pv1[22]*y**5 + \
        pv1[24]*x**6 + pv1[25]*x**5*y + pv1[26]*x**4*y**2 + pv1[27]*x**3*y**3 + \
        pv1[28]*x**2*y**4 + pv1[29]*x*y**5 + pv1[30]*y**6 + \
        pv1[31]*x**7 + pv1[32]*x**6*y + pv1[33]*x**5*y**2 + pv1[34]*x**4*y**3 + \
        pv1[35]*x**3*y**4 + pv1[36]*x**2*y**5 + pv1[37]*x*y**6 + pv1[38]*y**7
    tpvx = tpvx.expand()

    tpvy = pv2[0] + pv2[1]*y + pv2[2]*x + pv2[4]*y**2 + pv2[5]*y*x + pv2[6]*x**2 + \
        pv2[7]*y**3 + pv2[8]*y**2*x + pv2[9]*y*x**2 + pv2[10]*x**3 + \
        pv2[12]*y**4 + pv2[13]*y**3*x + pv2[14]*y**2*x**2 + pv2[15]*y*x**3 + pv2[16]*x**4 + \
        pv2[17]*y**5 + pv2[18]*y**4*x + pv2[19]*y**3*x**2 + pv2[20]*y**2*x**3 + pv2[21]*y*x**4 + \
        pv2[22]*x**5 + \
        pv2[24]*y**6 + pv2[25]*y**5*x + pv2[26]*y**4*x**2 + pv2[27]*y**3*x**3 + \
        pv2[28]*y**2*x**4 + pv2[29]*y*x**5 + pv2[30]*x**6 + \
        pv2[31]*y**7 + pv2[32]*y**6*x + pv2[33]*y**5*x**2 + pv2[34]*y**4*x**3 + \
        pv2[35]*y**3*x**4 + pv2[36]*y**2*x**5 + pv2[37]*y*x**6 + pv2[38]*x**7
    tpvy = tpvy.expand()
    return(pvrange, tpvx, tpvy)

def calcpv(pvrange,pvinx1, pvinx2, sipx, sipy, tpvx, tpvy):
    """Calculate the PV coefficient expression as a function of CD matrix
    parameters and SIP coefficients

    Parameters:
    -----------
    pvrange (list) : indices to the PV keywords
    pvinx1 (int): first index, 1 or 2
    pvinx2 (int): second index
    tpvx (Sympy expr) : equation for x-distortion in TPV convention
    tpvy (Sympy expr) : equation for y-distortion in TPV convention
    sipx (Sympy expr) : equation for x-distortion in SIP convention
    sipy (Sympy expr) : equation for y-distortion in SIP convention

    Returns:
    --------
    Expression of CD matrix elements and SIP polynomial coefficients
    """
    x, y = symbols("x y")
    if (pvinx1 == 1):
        expr1 = tpvx
        expr2 = sipx
    elif (pvinx1 == 2):
        expr1 = tpvy
        expr2 = sipy
    else:
        raise Valuerror('incorrect first index to PV keywords')
    if (pvinx2 not in pvrange):
        raise ValueError('incorrect second index to PV keywords')
    pvar = symbols('pv%d_%d'%(pvinx1, pvinx2))
    xord = yord = 0
    if expr1.coeff(pvar).has(x): xord = poly(expr1.coeff(pvar)).degree(x)
    if expr1.coeff(pvar).has(y): yord = poly(expr1.coeff(pvar)).degree(y)

    return(expr2.coeff(x,xord).coeff(y,yord))


def get_sip_keywords(header):
    """Return the  from a Header object

    Parameters:
    -----------
    header (pyfits.Header) : header object from a FITS file

    Returns:
    --------
    cd (numpy.matrix) : the CD matrix from the FITS header
    ac (numpy.matrix) : the A-coefficients from the FITS header
    bc (numpy.matrix) : the B-coefficients from the FITS header
    """
    dict = OrderedDict()
    cd = np.matrix([[header.get('CD1_1',0.0), header.get('CD1_2',0.0)],
                 [header.get('CD2_1',0.0), header.get('CD2_2',0.0)]], dtype=np.float64)
    ac = np.matrix(np.zeros((8,8), dtype=np.float64))
    bc = np.matrix(np.zeros((8,8), dtype=np.float64))
    for m in range(8):
        for n in range(0,8-m):
            ac[m,n] = header.get('A_%d_%d'%(m,n), 0.0)
            bc[m,n] = header.get('B_%d_%d'%(m,n), 0.0)
    return(cd, ac, bc)


def calc_sipexprs(cd, ac, bc):
    """ Calculate the Sympy expression for SIP distortion

    Parameters:
    -----------
    cd (numpy.matrix) : the CD matrix from the FITS header
    ac (numpy.matrix) : the A-coefficients from the FITS header
    bc (numpy.matrix) : the B-coefficients from the FITS header

    Returns:
    --------
    sipx (Sympy expr) : equation for x-distortion in SIP convention
    sipy (Sympy expr) : equation for y-distortion in SIP convention
    """
    x, y = symbols("x y")
    cdinverse = cd**-1
    invcd11 = cdinverse[0,0]
    invcd12 = cdinverse[0,1]
    invcd21 = cdinverse[1,0]
    invcd22 = cdinverse[1,1]
    uprime = invcd11*x+invcd12*y
    vprime = invcd21*x+invcd22*y
    usum = uprime
    vsum = vprime
    for m in range(8):
        for n in range(0,8-m):
            usum += ac[m,n]*uprime**m*vprime**n
            vsum += bc[m,n]*uprime**m*vprime**n
    sipx, sipy = cd*Matrix([usum, vsum])
    sipx = sipx.expand()
    sipy = sipy.expand()
    return(sipx, sipy)


def add_pv_keywords(header, sipx, sipy, pvrange, tpvx, tpvy, tpv=True):
    """Calculate the PV keywords and add to the header

    Parameters:
    -----------
    header (pyfits.Header) : header object from a FITS file
    tpv (boolean, default True) : Change CTYPE1/2 to TPV convention

    Returns:
    --------
    None (header is modified in place)
    """
    for p in pvrange:
        val = float(calcpv(pvrange,1,p,sipx,sipy, tpvx, tpvy).evalf())
        if val != 0.0:
            header['PV1_%d'%p] =  val
    for p in pvrange:
        val = float(calcpv(pvrange,2,p,sipx,sipy, tpvx, tpvy).evalf())
        if val != 0.0:
            header['PV2_%d'%p] = val
    if tpv:
        header['CTYPE1'] = 'RA---TPV'
        header['CTYPE2'] = 'DEC--TPV'
    else:
        header['CTYPE1'] = header['CTYPE1'][:8]
        header['CTYPE2'] = header['CTYPE2'][:8]
    return


def removekwd(header, kwd):
    """ Helper function for removing keywords from FITS headers after first
        testing that they exist in the header

    Parameters:
    -----------
    header (pyfits.Header) : header object from a FITS file
    kwd (string) : name of the keyword to be removed

    Returns:
    --------
    None (header is modified in place)
    """
    if kwd in header.keys():
        header.remove(kwd)
    return

def remove_sip_keywords(header):
    """ Remove keywords from the SIP convention from the header.

    Parameters:
    -----------
    header (pyfits.Header) : header object from a FITS file

    Returns:
    --------
    None (header is modified in place)
    """
    aorder = header.get('A_ORDER', 0)
    border = header.get('B_ORDER', 0)
    aporder = header.get('AP_ORDER', 0)
    bporder = header.get('BP_ORDER', 0)
    for m in range(aorder+1):
        for n in range(0,aorder+1-m):
            removekwd(header,'A_%d_%d'%(m,n))
    for m in range(border+1):
        for n in range(0,border+1-m):
            removekwd(header,'B_%d_%d'%(m,n))
    for m in range(aporder+1):
        for n in range(0,aporder+1-m):
            removekwd(header, 'AP_%d_%d'%(m,n))
    for m in range(bporder+1):
        for n in range(0,bporder+1-m):
            removekwd(header, 'BP_%d_%d'%(m,n))
    removekwd(header, 'A_ORDER')
    removekwd(header, 'B_ORDER')
    removekwd(header,'AP_ORDER')
    removekwd(header,'BP_ORDER')
    removekwd(header,'A_DMAX')
    removekwd(header,'B_DMAX')
    return


def sip_to_pv(infile,outfile,tpv_format=True,preserve=False,extension=0,clobber=True):
    """ Function which wraps the sip_to_pv conversion

    Parameters:
    -----------
    infile   (string) : name of input FITS file with TAN-SIP projection
    outfile (string) : name of output FITS file with TAN-TPV projection
    tpv_format (boolean) : modify CTYPE1 and CTYPE2 to TPV convention RA---TPV, DEC--TPV?
    preserve (boolean) : preserve the SIP keywords in the header (default is to delete)?
    extension (integer) : extension of FITS file containing SIP header (default 0)?

    Returns:
    --------
    True if TPV file has been created, False if not
    """
    hdu = pyfits.open(infile)
    header = hdu[extension].header
    pvrange, tpvx, tpvy = calc_tpvexprs()
    cd, ac, bc = get_sip_keywords(header)
    sipx, sipy = calc_sipexprs(cd, ac, bc)
    add_pv_keywords(header, sipx, sipy, pvrange, tpvx, tpvy, tpv=tpv_format)
    if (not preserve):
        remove_sip_keywords(header)
    hdu.writeto(outfile, clobber=clobber)

    if os.path.exists(outfile):
      return True
    else:
      return False

