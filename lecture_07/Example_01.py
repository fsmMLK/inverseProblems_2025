import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg


def generateSignal(n,showPlot=False,saveFig=False):
    """
    Generate a synthetic signal with specific characteristics.
    """
    x = np.linspace(0, n, n)
    f = np.zeros(n)
    n0 = round(n / 15)
    n1 = 2 * n0
    n2 = 3 * n0
    n3 = 4 * n0
    n4 = 8 * n0
    n5 = 10 * n0
    n6 = 14 * n0

    f[n1:n2] = 1
    f[n3:n4] = 1 + (np.arange(n3, n4) - n3) / (n4 - n3)
    f[n5:n6] = -1 - np.cos(np.arange(n5, n6) * 2 * np.pi / (n6 - n5))

    if showPlot:
        plt.figure()
        plt.plot(x, f, 'k', linewidth=2)
        plt.axis([0, n, -2, 3])
        plt.title('Original signal')
        plt.grid(True)
        if saveFig:
            plt.savefig('Ex01_image_signal.svg', format='svg', bbox_inches='tight')
        plt.show()

    return x, f

def generatePSF(n):
    PSF = np.array([1, 1, 1])
    PSF = np.convolve(PSF, PSF)
    PSF = np.convolve(PSF, PSF)
    PSF = np.convolve(PSF, PSF)
    PSF = PSF / np.sum(PSF)

    A = scipyLinalg.convolution_matrix(PSF, n, mode='same')

    return A,PSF

def generateMeasurements(A, f, delta=1e-2):
    m = A @ f
    noise = np.random.randn(len(f))
    noise = noise / np.linalg.norm(noise)
    md = m + delta * noise

    return m,md

if __name__ == "__main__":
    """# Step 1: Create signal, PSF and data"""
    n = 200
    x, f = generateSignal(n,showPlot=False)
    f-=5

    A,PSF = generatePSF(n)

    delta = 1e-3
    m,md = generateMeasurements(A,f,delta)

    """# Step 2: create the sample test for alpha"""
    alphaVec = np.logspace(-9, -2, 60)

    """# Step 3: If I knew the solution..."""
    error = np.zeros_like(alphaVec)
    for i,a in enumerate(alphaVec):
        fd = np.linalg.solve(A.T @ A + a * np.eye(n), A.T @ md)
        error[i] = np.linalg.norm(fd - f) / np.linalg.norm(f)

    indexMin = np.argmin(error)

    plt.figure(1)
    plt.loglog(alphaVec, error, 'bo-')
    plt.loglog(alphaVec[indexMin], error[indexMin], 'r.', markersize=20)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$|| T_\alpha f_{\alpha,\delta} - m_\delta ||$')
    plt.title(f'Real best one: %1.2g' % alphaVec[indexMin])
    plt.grid(True)
    plt.show()

    """# Step 4: Morozov discrepancy principle"""
    res = np.zeros_like(alphaVec)
    for i,a in enumerate(alphaVec):
        fd = np.linalg.solve(A.T @ A + a * np.eye(n), A.T @ md)
        res[i] = np.linalg.norm(A @ fd - md)

    indexMin = np.argmin(np.abs(res - delta))

    plt.figure(2)
    plt.loglog(alphaVec, res, 'bo-')
    plt.title(f'alpha Morozov: %1.2g' % alphaVec[indexMin])
    plt.loglog(alphaVec, delta * np.ones_like(alphaVec), '--k')
    plt.loglog(alphaVec[indexMin], res[indexMin], 'r.', markersize=20)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$|| T_\alpha f_{\alpha,\delta} - m_\delta ||$')
    plt.grid(True)
    plt.show()

    """# Step 4: L curve"""
    X = np.zeros_like(alphaVec)
    Y = np.zeros_like(alphaVec)
    for i,a in enumerate(alphaVec):
        fd = np.linalg.solve(A.T @ A + a * np.eye(n), A.T @ md)
        X[i] = np.log(np.linalg.norm(A @ fd - md))
        Y[i] = np.log(np.linalg.norm(fd))

    plt.figure(3)

    # Rescale X and Y
    Xn = (X - np.min(X)) / (np.max(X) - np.min(X))
    Yn = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    indexMin = np.argmin(Xn**2 + Yn**2)  # find the closest point to the origin

    plt.plot(X, Y, 'bo-')
    plt.title(f'alpha L curve: %1.2g' % alphaVec[indexMin])
    plt.plot(X[indexMin], Y[indexMin], 'r.', markersize=25)
    plt.xlabel(r'$\log(|| T_\alpha f_{\alpha,\delta} - m_\delta ||)$')
    plt.ylabel(r'$\log(|| f_{\alpha,\delta} ||)$')
    plt.grid(True)
    plt.show()