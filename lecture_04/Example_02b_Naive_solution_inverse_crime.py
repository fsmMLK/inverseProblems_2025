import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipyIntegrate
import scipy.interpolate as scipyInterpolate
import scipy.linalg as scipyLinalg

# setting font size to 20
plt.rcParams.update({'font.size': 20})

def createTarget(x):
    """
    This file defines a piecewise linear target function on the interval
    [0,1]. The function will be used in the one-dimensional deconvolution
    example.

    Arguments:
    x   vector of real numbers. Can be a single number or a numpy array.

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


def createPSF(x, a=0.04):
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


def convolveContinuousModel(flagPlot=True):
    """
    This function computes the convolution of a piecewise linear target function
    with a point spread function (PSF) defined by a polynomial. The target function
    is periodic and defined on the interval [0, 1]. The PSF is defined on the interval
    [-a, a] where a is a small positive constant. The convolution is computed by
    integrating the product of the PSF and the shifted target function over the
    interval [0, 1].

    Here 'continuum' is in quotation marks because it is not really continuum
    but rather very finely sampling compared to the samplings used in
    computational inversion.

    :param flagPlot: If True, the results will be plotted.
    :type flagPlot: bool

    :return:
    This function returns a dictionary with the following keys
    - 'xSignal': The points at which the target function is evaluated.
    - 'signal': The values of the target function at the points in xSignal.
    - 'convSignal': The convolved signal obtained by integrating the product of the PSF and the shifted target function.
    - 'xPSF': The points at which the PSF is evaluated.
    - 'PSF': The values of the point spread function at the points in xPSF.
    - 'a': The parameter a used in the PSF definition.
    :rtype: dict

    :raises ValueError: If the parameter a is not in the acceptable range (0 < a < 1/2).

    Translated from MATLAB code by Jennifer Mueller and Samuli Siltanen, October 2012

    Translated to Python by Fernando moura, July 2025
    """

    # ---------------------------------------------------------------
    # target function is defined on the interval [0,1]
    # ---------------------------------------------------------------

    # Choose the 'continuum' points at which to compute the convolved function.
    sizeSignal = 2000
    xSignal = np.linspace(0, 1, sizeSignal)
    signal = createTarget(xSignal)

    # ---------------------------------------------------------------
    # point spread function (PSF) is defined on the interval [-a,a], 0<a<1/2
    # ---------------------------------------------------------------

    # Parameter that specifies the width of the PSF, must satisfy 0<a<1/2.
    # The support of the building block of the PSF is [-a,a], and this
    # building block is replicated at each integer to produce a periodic function.
    a = 0.04
    # Create numerical integration points. We take quite a fine sampling here
    # to ensure accurate approximation of the convolution integral.
    sizePSF = 1000
    xPSF = np.linspace(-a, a, sizePSF)
    deltaXpsf = xPSF[1] - xPSF[0]

    # Evaluate normalized PSF at integration points
    PSF = np.zeros_like(xPSF)
    ind = np.abs(xPSF) < a
    PSF[ind] = createPSF(xPSF[ind], a)
    PSF = PSF / scipyIntegrate.trapezoid(PSF, dx=deltaXpsf)  # Normalization

    # ---------------------------------------------------------------
    # compute the convolution integral
    # ---------------------------------------------------------------

    # Initialize result
    convolvedSignal = np.zeros_like(xSignal)

    # Compute convolution by integration using trapezoidal rule
    for i in range(sizeSignal):
        # Shift the target function to match the current point
        # This is the convolution integral: Af[i] = integral_0^1 psf(x) * target(x - xSignal[i]) dx
        targ = createTarget(xSignal[i] - xPSF)

        convolvedSignal[i] = scipyIntegrate.trapezoid(PSF * targ, dx=deltaXpsf)

    # Plot the results
    if flagPlot:
        plt.figure(figsize=(10, 6))
        plt.plot(xSignal, signal, 'k-', label='Target Function')
        plt.plot(xSignal, convolvedSignal, 'r-', label='Convolved Function')
        plt.title('Convolution Result (Continuous model)')
        plt.xlabel('x')
        plt.ylabel('signal')
        plt.legend()
        plt.grid(True)
        plt.show()

    return {'xSignal': xSignal, 'signal': signal, 'convSignal': convolvedSignal, 'xPSF': xPSF, 'PSF': PSF, 'a': a}


def convolveDiscreteModel(continuousDataDict, flagPlot=True):
    """
    Simulate 1-dimensional convolution data that does not involve inverse crime.
    This is done by computing data using a very fine discretization and a perturbed
    point spread function. That data is then sampled at the desired computational grid,
    and random noise is added.

    :param continuousDataDict: Dictionary containing precomputed continuous model data.
    :type continuousDataDict: dict
    :param flagPlot: If True, the results will be plotted.
    :type flagPlot: bool

    :return:
    This function returns a dictionary with the following keys
    - 'xSignal': The points at which the target function is evaluated.
    - 'm': The convolved signal without noise.
    - 'm_noise': The convolved signal with added noise.
    - 'm_invCrime': The convolved signal computed with inverse crime.
    - 'noiseSigma': The standard deviation of the noise added to the convolved signal.
    - 'A': The convolution matrix used for the inverse crime computation.

    :rtype: dict
    :raises ValueError: If the parameter a is not in the acceptable range (0 < a < 1/2).

    Translated from MATLAB code by Jennifer Mueller and Samuli Siltanen, October 2012
    Translated to Python by Fernando moura, July 2025
    """

    # Construct discretization points
    n = 64;
    x_discrete = np.linspace(0, 1, n)
    Dx = x_discrete[1] - x_discrete[0]

    # Choose two noise levels. We compute three kinds of data: no added noise,
    # and noisy data with noise amplitude given by sigma
    sigma = 0.05

    # Load precomputed results from continuous model
    f_cont = continuousDataDict['signal'].flatten()
    x_cont = continuousDataDict['xSignal'].flatten()
    Af_cont = continuousDataDict['convSignal'].flatten()
    a = continuousDataDict['a']

    # -------------------------------------------------------
    # COMPUTE CONVOLUTION DATA WITH ->>NO INVERSE CRIME <<-
    # -------------------------------------------------------

    # Note that we are taking samples of the continuous convolution result and
    # the deconvolution will be done with a discrete aproximation of the
    # convolution operator.

    # Interpolate values of the convolution at points x using the precomputed values on the fine
    # grid called x_cont
    interpFunc = scipyInterpolate.CubicSpline(x_cont, Af_cont, bc_type='periodic')

    # Create data without noise
    m = interpFunc(x_discrete)

    # Create data with random measurement noise and no inverse crime
    noise = sigma * np.max(np.abs(m)) * np.random.randn(len(m))
    mn = m + noise

    # Compute the amount of simulated measurement noise in mn
    relerr = np.max(np.abs(m - mn)) / np.max(np.abs(m))
    relerr2 = np.linalg.norm(m - mn) / np.linalg.norm(m)
    print(f'Relative sup norm error in mn is {relerr:.4f}')
    print(f'Relative square norm error in mn is {relerr2:.4f}')

    # -------------------------------------------------------
    # CONSTRUCT MATRIX MODEL AND COMPUTE DATA ->> WITH INVERSE CRIME <<-
    # -------------------------------------------------------

    nPSF = int(np.ceil(a / Dx))
    xPSF = np.arange(-nPSF, nPSF + 1) * Dx
    PSF = np.zeros_like(xPSF)
    ind = np.abs(xPSF) < a
    PSF[ind] = createPSF(xPSF[ind], a)
    PSF = PSF / scipyIntegrate.trapezoid(PSF, dx=Dx)  # Normalization

    # Construct convolution matrix
    A = Dx * scipyLinalg.convolution_matrix(PSF, n, mode='same')

    # Compute ideal data WITH INVERSE CRIME
    f = createTarget(x_discrete)
    mIC = A @ f

    # Plot results
    if flagPlot:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        axes[0].plot(x_discrete, m, 'bo', label='Discrete')
        axes[0].set_title('Convolution data')

        axes[1].plot(x_discrete, mn, 'bo', label='Discrete ')

        axes[2].plot(x_discrete, mIC, 'bo', label='Discrete')

        for ax in axes:
            ax.plot(x_cont, f_cont, 'k-', label='Target Function')
            ax.plot(x_cont, Af_cont, 'r-', label='Continuous')
            ax.grid(True)

        axes[0].legend(loc='upper left', bbox_to_anchor=(0.85, 0.95))
        axes[0].text(0.5, 1.25, 'no noise, no inverse crime', dict(size=20), ha='center')
        axes[1].text(0.5, 1.25, 'with %d%% noise, no inverse crime' % (sigma * 100), dict(size=20), ha='center')
        axes[2].text(0.5, 1.25, 'with inverse crime', dict(size=20), ha='center')
        plt.tight_layout()
        plt.show()

    return {'xSignal': x_discrete, 'm': m, 'm_noise': mn, 'm_invCrime': mIC, 'noiseSigma': sigma,'A':A}


def deconvolveNaive(continuousData,discretedata):
    """
    This function performs naive deconvolution using the convolution matrix A
    and the noisy data mn.
    """

    A = discretedata['A']
    x_discrete = discretedata['xSignal']
    n = x_discrete.shape[0]
    m = discretedata['m']
    mn = discretedata['m_noise']
    mIC = discretedata['m_invCrime']
    f_cont = continuousData['signal']
    x_cont = continuousData['xSignal']
    sigma = discretedata['noiseSigma']

    # Compute reconstruction from data m, mn, mIC
    reco = scipyLinalg.inv(A) @ m      # without inverse crime, no noise
    reco_noise = scipyLinalg.inv(A) @ mn # with inverse crime noise
    reco_IC = scipyLinalg.inv(A) @ mIC # with inverse crime

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(x_discrete, reco, 'ro', label='Deconvolution')
    axes[0].set_title('Convolution data')
    axes[1].plot(x_discrete, reco_noise, 'ro', label='Deconvolution ')
    axes[2].plot(x_discrete, reco_IC, 'ro', label='Deconvolution')

    for ax in axes:
        ax.plot(x_cont, f_cont, 'k-', label='Target Function')
        ax.set_ylim([-0.1, 1.6])
        ax.grid(True)

    axes[0].legend(loc='upper left', bbox_to_anchor=(0.85, 0.95))
    axes[0].text(0.5, 1.25, 'no noise, no inverse crime', dict(size=20), ha='center')
    axes[1].text(0.5, 1.25, 'with %d%% noise, no inverse crime' % (sigma * 100), dict(size=20), ha='center')
    axes[2].text(0.5, 1.25, 'with inverse crime', dict(size=20), ha='center')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    continuousData = convolveContinuousModel(flagPlot=False)
    discreteData = convolveDiscreteModel(continuousData, flagPlot=True)
    deconvolveNaive(continuousData,discreteData)
