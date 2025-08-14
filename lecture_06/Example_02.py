import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg
from Example_01 import generateSignal

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
    """# Step 1: Create signal"""
    n = 100
    x, f = generateSignal(n,showPlot=False)
    f+=5

    """# Step 2: point spread function and convolution matrix"""
    A,PSF = generatePSF(n)

    """# Step 3: create data"""
    delta = 1e-2
    m,md = generateMeasurements(A,f,delta)

    """# Step 4: Tikhonov"""
    alpha = 0.1
    fd = np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ md)

    plt.figure()
    plt.plot(x, f, 'k', label='Original signal')
    plt.plot(x, md, 'r', label='noisy data')
    plt.plot(x, fd, 'b', label='Tikhonov solution')
    plt.legend()
    plt.axis([0, n, 0, 8])
    plt.grid(True)
    plt.show()

    """# Step 5: Generalized Tikhonov"""
    f0 = 5 * np.ones(n)  # initial guess
    fd = np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ md + alpha * f0)

    plt.figure()
    plt.plot(x, f, 'k', label='Original signal')
    plt.plot(x, md, 'r', label='noisy data')
    plt.plot(x, fd, 'b', label='Generalized Tikhonov')
    plt.legend()
    plt.axis([0, n, 0, 8])
    plt.grid(True)
    plt.show()
