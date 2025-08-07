import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg

def generateSignal(n):
    """
    Generate a synthetic signal with specific characteristics.
    """
    x = np.linspace(0, n, n)
    f = 5 * np.ones(n)
    n0 = round(n / 15)
    n1 = 2 * n0
    n4 = 8 * n0
    n5 = 10 * n0
    n6 = 14 * n0

    f[n1:n4] = 6
    f[n5:n6] = 4 - np.cos(np.arange(n5, n6) * 2 * np.pi / (n6 - n5))

    return x, f

"""# Step 1: Create signal"""
n = 100
x, f = generateSignal(n)
f-=5

plt.figure()
plt.plot(x, f, 'k', linewidth=2)
plt.axis([0, n, 0, 7])
plt.grid(True)
plt.show()

"""# Step 2: point spread function and convolution matrix"""
PSF = np.array([1, 1, 1])
PSF = np.convolve(PSF, PSF)
PSF = np.convolve(PSF, PSF)
PSF = np.convolve(PSF, PSF)
PSF = PSF / np.sum(PSF)

A = scipyLinalg.convolution_matrix(PSF, n, mode='same')

"""# Step 3: create data"""
m = A @ f
delta = 1e-2
noise = np.random.randn(n)
noise = noise / np.linalg.norm(noise)
md = m + delta * noise

"""# Step 4: Tikhonov"""
alpha = 0.1
fd = np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ md)

plt.figure()
plt.plot(x, f, 'k', label='Original signal')
plt.plot(x, md, 'r', label='noisy data')
plt.plot(x, fd, 'b', label='Tikhonov solution')
plt.legend()
plt.axis([0, n, 0, 7])
plt.grid(True)
plt.show()

"""# Step 5: Generalized Tikhonov"""
L = np.eye(n) - np.diag(np.ones(n-1), -1)  # the matrix representing the derivative
fd = np.linalg.solve(A.T @ A + alpha * (L.T @ L), A.T @ md)

plt.figure()
plt.plot(x, f, 'k', label='Original signal')
plt.plot(x, md, 'r', label='noisy data')
plt.plot(x, fd, 'b', label='Generalized Tikhonov')
plt.legend()
plt.axis([0, n, -3, 2])
plt.grid(True)
plt.show()

