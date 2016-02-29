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
    pvrange = range(0,39)
    pvrange.remove(3)
    pvrange.remove(11)
    pvrange.remove(23)
    for k in 1,2:
        for p in pvrange:
            name = "pv%d_%d" % (k,p)
            exec('%s = symbols("%s")'%(name,name))

    # Copy the equations from the PV-to-SIP paper and convert to code,
    #  leaving out radial terms PV[1,3], PV[1,11], PV[1,23], PV[1,39]

    tpvx = pv1_0 + pv1_1*x + pv1_2*y + pv1_4*x**2 + pv1_5*x*y + pv1_6*y**2 + \
        pv1_7*x**3 + pv1_8*x**2*y + pv1_9*x*y**2 + pv1_10*y**3 +  \
        pv1_12*x**4 + pv1_13*x**3*y + pv1_14*x**2*y**2 + pv1_15*x*y**3 + pv1_16*y**4 + \
        pv1_17*x**5 + pv1_18*x**4*y + pv1_19*x**3*y**2 + pv1_20*x**2*y**3 + pv1_21*x*y**4 + \
        pv1_22*y**5 + \
        pv1_24*x**6 + pv1_25*x**5*y + pv1_26*x**4*y**2 + pv1_27*x**3*y**3 + \
        pv1_28*x**2*y**4 + pv1_29*x*y**5 + pv1_30*y**6 + \
        pv1_31*x**7 + pv1_32*x**6*y + pv1_33*x**5*y**2 + pv1_34*x**4*y**3 + \
        pv1_35*x**3*y**4 + pv1_36*x**2*y**5 + pv1_37*x*y**6 + pv1_38*y**7
    tpvx = tpvx.expand()

    tpvy = pv2_0 + pv2_1*y + pv2_2*x + pv2_4*y**2 + pv2_5*y*x + pv2_6*x**2 + \
        pv2_7*y**3 + pv2_8*y**2*x + pv2_9*y*x**2 + pv2_10*x**3 + \
        pv2_12*y**4 + pv2_13*y**3*x + pv2_14*y**2*x**2 + pv2_15*y*x**3 + pv2_16*x**4 + \
        pv2_17*y**5 + pv2_18*y**4*x + pv2_19*y**3*x**2 + pv2_20*y**2*x**3 + pv2_21*y*x**4 + \
        pv2_22*x**5 + \
        pv2_24*y**6 + pv2_25*y**5*x + pv2_26*y**4*x**2 + pv2_27*y**3*x**3 + \
        pv2_28*y**2*x**4 + pv2_29*y*x**5 + pv2_30*x**6 + \
        pv2_31*y**7 + pv2_32*y**6*x + pv2_33*y**5*x**2 + pv2_34*y**4*x**3 + \
        pv2_35*y**3*x**4 + pv2_36*y**2*x**5 + pv2_37*y*x**6 + pv2_38*x**7
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
        raise Valuerror, 'incorrect first index to PV keywords'
    if (pvinx2 not in pvrange):
        raise ValueError, 'incorrect second index to PV keywords'
    exec("pvar = 'pv%d_%d'"%(pvinx1, pvinx2))
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
    aorder = int(header.get('A_ORDER', 0.0))
    border = int(header.get('B_ORDER', 0.0))
    aporder = int(header.get('AP_ORDER', 0.0))
    bporder = int(header.get('BP_ORDER', 0.0))
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

