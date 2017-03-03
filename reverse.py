
from __future__ import print_function, division, absolute_import
# This code determines reverse coefficients from the forward ones
import numpy as np


# function to evaluate polynomials
def evalpoly(u, v, adist, bdist):
    """ Given coordinates arrays u, v,
        polynomial orders aorder and border,
        and distortion polynomial matrices,
        return a tuple of (uprime,vprime) containing the corrected positions."""

    # Calculate forward orders from size of distortion matrices
    aorder = np.shape(adist)[0] - 1
    border = np.shape(bdist)[0] - 1
    myshape = u.shape
    uu = u.flatten()
    vv = v.flatten()
    uprime = uu.astype('float64')
    vprime = vv.astype('float64')

    udict = {}
    vdict = {}
    for i in range(max(aorder,border)+1):
        udict[i] = np.power(uu,i)
        vdict[i] = np.power(vv,i)
    i = aorder
    while (i >= 0):
        j = aorder - i
        while (j >= 0):
            uprime += adist[i][j]*udict[i]*vdict[j]
            j -= 1
        i -= 1
    i = border
    while (i >= 0):
        j = border - i
        while (j >= 0):
            vprime += bdist[i][j]*udict[i]*vdict[j]
            j -= 1
        i -= 1
    uprime = uprime.reshape(myshape)
    vprime = vprime.reshape(myshape)
    return (uprime, vprime)


def fitreverse(aporder, bporder, adist, bdist, u, v):
    """ Given the desired reverse polynomials orders,
        the forward coefficients, and coordinate arrays
        u and v for doing the calculations, this function
        computes reverse coefficients and returns the results
        in matrices apdist and bpdist """

    # Create reverse coefficient matrices
    apdist = np.zeros((aporder+1,aporder+1),'float64')
    bpdist = np.zeros((bporder+1,bporder+1),'float64')

    (uprime, vprime) = evalpoly(u,v,adist,bdist)
    updict = {}
    vpdict = {}
    for i in range(max(aporder,bporder)+5):
        updict[i] = np.power(uprime.flatten(),i)
        vpdict[i] = np.power(vprime.flatten(),i)
    udiff = u - uprime
    vdiff = v - vprime

    mylist1 = []
    mylist2 = []
    for i in range(aporder+1):
        for j in range(0, aporder - i + 1):
            mylist1.append(updict[i]*vpdict[j])
    for i in range(bporder+1):
        for j in range(0, bporder - i + 1):
            mylist2.append(updict[i]*vpdict[j])

    A = np.array(mylist1).T
    B = udiff.flatten()

    apcoeffs, r, rank, s = np.linalg.lstsq(A, B)

    A = np.array(mylist2).T
    B = vdiff.flatten()

    bpcoeffs, r, rank, s = np.linalg.lstsq(A, B)

    # Load reverse distortion matrices
    extractcoeffs(apcoeffs, aporder, apdist)
    extractcoeffs(bpcoeffs, bporder, bpdist)

    return(apdist, bpdist)


def extractcoeffs(coeffs, order, dist):
    """ Given a compact vector of coefficients and the
        polynomial order, extract them into matrix dist """
    index = 0
    for i in range(order+1):
        for j in range(0,order-i+1):
            dist[i][j] = coeffs[index]
            index += 1


