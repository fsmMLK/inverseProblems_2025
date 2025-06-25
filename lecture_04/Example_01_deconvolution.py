import numpy as np
import matplotlib.pyplot as plt


def DC_target(x):
    """
    This file defines a piecewise linear target function on the interval
    [0,1]. The function will be used in the one-dimensional deconvolution
    example.

    Arguments:
    x   vector of real numbers

    Returns: values of f at the points specified in argument vector x. Note
    that the target function is thought to be 1-periodic; if the argument
    vector contains points outside the interval [0,1], then periodicity is
    used to evaluate f.

    Translated from MATLAB code by Jennifer Mueller and Samuli Siltanen, October 2012
    """

    # Initialize result
    f = np.zeros_like(x)

    # Enforce periodicity by taking only decimal part of each real number
    x = x - np.floor(x)

    # Set values of f wherever it does not equal zero
    f[(x >= 0.12) & (x <= 0.15)] = 1.5
    f[(x >= 0.2) & (x <= 0.25)] = 1.3
    f[(x >= 0.75) & (x <= 0.85)] = 1
    ind = (x >= 0.35) & (x <= 0.55)
    f[ind] = 5 * (x[ind] - 0.35)

    return f


def DC_PSF(x, a):
    """
    Point spread function used in one-dimensional deconvolution example.
    The function differs from zero only in the interval [-a, a] with a > 0.
    Defined by the formula: (x + a)^2 * (x - a)^2

    This routine does not involve any normalization of the point spread function.

    Arguments:
    x -- array of evaluation points (real numbers)
    a -- positive constant, should satisfy 0 < a < 1/2

    Raises:
    ValueError if parameter a is not in acceptable range

    Translated from MATLAB code by Jennifer Mueller and Samuli Siltanen, October 2012
    """

    # Check that parameter a is in acceptable range
    if a <= 0 or a >= 1 / 2:
        raise ValueError('Parameter a should satisfy 0 < a < 1/2')

    # Evaluate the polynomial
    psf = (x + a) ** 2 * (x - a) ** 2

    return psf

def DC_PSF_plot():

    # Plot parameters
    fsize = 20
    lwidth = 1.5

    # Parameter that specifies the width of the PSF, must satisfy 0<a<1/2
    a = 0.1

    ################################################################
    # First plot: the building block only

    # Create plot points
    Nxx = 512
    xx = np.linspace(-0.5, 0.5, Nxx)
    Dxx = xx[1] - xx[0]

    # Evaluate point spread function
    psf = np.zeros_like(xx)
    ind = np.abs(xx) < a
    psf[ind] = DC_PSF(xx[ind], a)
    Ca = 1 / (Dxx * np.sum(np.abs(psf)))  # Normalization constant
    psf = Ca * psf

    # Create plot window
    plt.figure(1)
    plt.clf()

    # Plot the psf
    plt.plot(xx, psf, 'k', linewidth=lwidth)
    plt.xlim([-1/2, 1/2])
    plt.box(False)
    plt.gca().set_aspect(2)  # Equivalent to PlotBoxAspectRatio [2 1 1]

    ################################################################
    # Second plot: the periodic psf

    # Create plot points
    Nxx2 = 2048
    xx2 = np.linspace(-np.pi/2, 2.5*np.pi, Nxx2)
    Dxx2 = xx2[1] - xx2[0]

    # Evaluate point spread function
    psf2 = np.zeros_like(xx2)
    ind2 = (np.abs(xx2) < a)
    ind3 = (np.abs(xx2 - 1) < a)
    ind4 = (np.abs(xx2 + 1) < a)
    psf2[ind2] = DC_PSF(xx2[ind2], a)
    psf2[ind3] = DC_PSF(xx2[ind3] - 1, a)
    psf2[ind4] = DC_PSF(xx2[ind4] + 1, a)
    psf2 = Ca * psf2

    # Create plot window
    plt.figure(2)
    plt.clf()

    # Plot the psf
    plt.plot(xx2, psf2, 'k', linewidth=lwidth)
    plt.axis([-1.5, 1.5, 0, 25])
    plt.box(False)
    plt.yticks([0, 10, 20, 23.4], fontsize=fsize)
    plt.xticks([-1, 0, 1], fontsize=fsize)
    plt.gca().set_aspect(2)

    # Show the plots
    plt.show()