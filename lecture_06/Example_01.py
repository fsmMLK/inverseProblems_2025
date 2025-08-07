from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg

def generateSignal(n):
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
    f[n3:n4] = 3 * (np.arange(n3, n4) - n3) / (n4 - n3)
    f[n5:n6] = -1 - np.cos(np.arange(n5, n6) * 2 * np.pi / (n6 - n5))

    return x,f

"""# Step 1: Create signal, psf and convolution matrix"""
n=100
x,f = generateSignal(n)

plt.figure()
plt.plot(x, f, 'k', linewidth=2)
plt.axis([0, n, -2, 3])
plt.title('Original signal')
plt.grid(True)
plt.show()

psf = np.array([1, 4, 8, 16, 19, 15, 10, 7, 1])
psf = psf / np.sum(psf)

# convolution matrix
A = scipyLinalg.convolution_matrix(psf, n, mode='same')

"""# Step 2: noisy measurement"""
m = A @ f
delta = 1e-2
noise = np.random.randn(n)
noise = noise / np.linalg.norm(noise)
md = m + delta * noise

# Step 3: Tikhonov
alpha = 0.1

# Strategy I: SVD
tic = time()
U, svals, Vt = scipyLinalg.svd(A)
svals_tik = np.diag(svals / (svals ** 2 + alpha))
T_a = Vt.T @ svals_tik @ U.T
fd1 = T_a @ md
t1 = time() - tic

# Strategy II: normal equations
tic = time()
fd2 = np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ md)
t2 = time() - tic

# Strategy III: stacked form
tic = time()
A_stack = np.vstack([A, np.sqrt(alpha) * np.eye(n)])
md_stack = np.concatenate([md, np.zeros(n)])
fd3 = np.linalg.lstsq(A_stack, md_stack, rcond=None)[0]
t3 = time() - tic

# 4 Comparison

plt.figure()
plt.plot(x, f, 'k', label='Original signal')
plt.plot(x, md, 'r', label='noisy data')
plt.plot(x, fd1, 'b', label='SVD-based')
plt.plot(x, fd2, 'm', label='normal equations')
plt.plot(x, fd3, 'g', label='stacked form')
plt.legend()
plt.axis([0, n, -2, 3])
plt.grid(True)
plt.show()

print(np.linalg.norm(fd1 - fd2))
print(np.linalg.norm(fd1 - fd3))
print(f'Singular values : {t1:.5f} s')
print(f'Normal equations: {t2:.5f} s')
print(f'Stacked formula : {t3:.5f} s')
