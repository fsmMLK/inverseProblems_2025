import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg
from Example_01 import generateSignal
from Example_02 import generatePSF,generateMeasurements

def generateDiffOperator(size):
    """
    Generate a finite difference operator for 1D signals.
    """
    D = np.zeros((size - 1, size))
    for i in range(size - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D

if __name__ == "__main__":
    """# Step 1: Create signal"""
    n = 200
    x, f = generateSignal(n,showPlot=False)


    """# Step 2: point spread function and convolution matrix"""
    A,PSF = generatePSF(n)

    """# Step 3: create data"""
    delta = 1e-2
    m,md = generateMeasurements(A,f,delta=1e-2)

    """# Step 4: Tikhonov"""
    alpha = 0.1
    fd = np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ md)

    """# Step 5: Generalized Tikhonov"""
    L = generateDiffOperator(n)
    fdG = np.linalg.solve(A.T @ A + alpha * (L.T @ L), A.T @ md)

    plt.figure()
    plt.plot(x, f, 'k', label='Original signal')
    plt.plot(x, md, 'r', label='noisy data')
    plt.plot(x, fd, 'g', label='Tikhonov solution')
    plt.plot(x, fdG, 'b', label='Generalized Tikhonov')
    plt.legend()
    plt.axis([0, n, -3, 3])
    plt.grid(True)
    plt.show()

