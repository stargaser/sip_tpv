
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

    uprime = u.astype('float64')
    vprime = v.astype('float64')
    i = aorder
    while (i >= 0):
        j = aorder - i
        while (j >= 0):
            uprime += adist[i][j]*(u**i)*(v**j)
            j -= 1
        i -= 1
    i = border
    while (i >= 0):
        j = border - i
        while (j >= 0):
            vprime += bdist[i][j]*(u**i)*(v**j)
            j -= 1
        i -= 1
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
    udiff = u - uprime
    (uvec, umat) = reversesetup(udiff, aporder, uprime, vprime)
    vdiff = v - vprime
    (vvec, vmat) = reversesetup(vdiff, bporder, uprime, vprime)

    # Now at last, we are ready for inversion
    uinv = np.linalg.inv(umat)
    vinv = np.linalg.inv(vmat)

    apcoeffs = np.dot(uinv, uvec)
    bpcoeffs = np.dot(vinv, vvec)

    # Load reverse distortion matrices
    extractcoeffs(apcoeffs, aporder, apdist)
    extractcoeffs(bpcoeffs, bporder, bpdist)

    return(apdist, bpdist)

def reversesetup(pdiff, order, x, y):
    """ Given pixel difference array ,order,
        and x & y coord arrays,
        return elements in vector vec and matrix mat
        to set up least-squares solution for reverse
        polynomial coefficients """

    # This number gives the array dimensions (number of coefficients)
    idim = (order*(order+3))/2

    # Create vector and matrix for calculations
    vec = np.zeros((idim,), 'float64')
    mat = np.zeros((idim, idim), 'float64')
    sumpdiff = np.sum(pdiff, axis=None)

    index = -1
    for i in range(order+1):
        for j in range(order-i+1):
            if (index >= 0):
                vec[index] = sumpdiff*np.sum(np.power(x,i)*np.power(y,j), axis=None)
                l = -1
                for xpow in range(order+1):
                    for ypow in range(order-xpow+1):
                        if (l >= 0):
                            mat[l][index] = \
                                    np.sum(np.power(x,i+xpow)*np.power(y,j+ypow),
                                                  axis=None)
                        l += 1
            index += 1
    return(vec, mat)

def extractcoeffs(coeffs, order, dist):
    """ Given a compact vector of coefficients and the
        polynomial order, extract them into matrix dist """
    index = -1
    for i in range(order+1):
        for j in range(order-i+1):
            if (index >= 0):
                dist[i][j] = coeffs[index]
            index += 1


