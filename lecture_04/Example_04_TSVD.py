import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scipyLinalg

# This file is for experimenting with deconvolution with two different signals.

# Step 1: create two different signals f1 and f2, one smooth and one spiky.
# Choose the dimension of the unknown vector
N = 256

# Construct evaluation points
x = np.linspace(0, 2*np.pi, N).reshape(-1, 1)

# Construct smooth signal f1
f1 = (2*np.pi - x) * x**3 * (1 - np.cos(x))
f1 = f1 / np.max(f1)

# Construct spiky signal f2
f2 = np.zeros((N, 1))
f2[round(N/5)] = 1
f2[round(10*N/23) + np.arange(1, 3)] = 0.3
f2[round(3*N/4)] = 0.7
f2[round(5*N/6) + np.arange(1, 3)] = 0.7

# Take a look at the signals
plt.figure(1)
plt.subplot(121)
plt.plot(f1, 'k')
plt.axis([1, N, 0, 1.1])
plt.title('Smooth signal f_1')
plt.subplot(122)
plt.plot(f2, 'k')
plt.axis([1, N, 0, 1.1])
plt.title('Spiky signal f_2')
plt.show()

# Step 2: Construct the PSF (Point Spread Function) for the convolution and convolution matrix.
PSF = np.array([1, 1, 1])
for i in range(3):
    PSF = np.convolve(PSF, PSF) # this ensures the kernel is smooth

# Normalize the PSF
PSF = PSF / np.sum(PSF)

# Take a look
plt.figure(2)
plt.clf()
plt.plot(PSF, 'r')
plt.plot(PSF, 'r.', markersize=12)
plt.title(f'Convolution kernel of length {len(PSF)}')
plt.show()

#convolution matrix
A = scipyLinalg.convolution_matrix(PSF, N, mode='same')

# Step 3: Compute SVD
U, svals, Vh = scipyLinalg.svd(A, full_matrices=False)
V = Vh.T
S = np.diag(svals)

# Take a look
plt.figure(3)
plt.clf()
plt.subplot(121)
plt.spy(A)
plt.title(f'Convolution matrix of size {A.shape[0]}x{A.shape[1]}')
plt.subplot(122)
plt.semilogy(svals)
plt.xlim([1, N])
plt.title('Singular values')
plt.show()

# Step 4: Construct convolved signals and noisy data, including inverse crime
# The "data," with inverse crime
m1 = A @ f1
m2 = A @ f2

# Noisy data
mn1 = m1 + 0.01 * np.random.randn(*m1.shape)
mn2 = m2 + 0.01 * np.random.randn(*m2.shape)

# Take a look
plt.figure(4)
plt.clf()
plt.subplot(121)
plt.plot(f1, 'k')
plt.plot(mn1, 'r')
plt.axis([1, N, 0, 1.1])
plt.title('Noisy data of signal 1')
plt.subplot(122)
plt.plot(f2, 'k')
plt.plot(mn2, 'r')
plt.axis([1, N, 0, 1.1])
plt.title('Noisy data of signal 2')
plt.show()

# Step 5: Reconstruct the signals using naive inversion
rec1 = scipyLinalg.inv(A) @ mn1
rec2 = scipyLinalg.inv(A) @ mn2

# Take a look
plt.figure(4)
plt.clf()
plt.subplot(121)
plt.plot(f1, 'k')
plt.plot(rec1, 'b')
plt.xlim([1, N])
plt.subplot(122)
plt.plot(f2, 'k')
plt.plot(rec2, 'b')
plt.xlim([1, N])
plt.show()

# Step 6: Reconstruct the signals using truncated SVD (TSVD)
r_alpha = 64
Sralpha = np.zeros(S.T.shape)
svals_ralpha = svals[:r_alpha]
Sralpha[:r_alpha, :r_alpha] = np.diag(1 / svals_ralpha)

# Reconstruct with TSVD
TSVDrec1 = V @ Sralpha @ U.T @ mn1
TSVDrec2 = V @ Sralpha @ U.T @ mn2

# Take a look
plt.figure(5)
plt.clf()
plt.subplot(121)
plt.plot(f1, 'k')
plt.plot(TSVDrec1, 'r')
plt.xlim([1, N])
plt.subplot(122)
plt.plot(f2, 'k')
plt.plot(TSVDrec2, 'r')
plt.xlim([1, N])
plt.show()

# Take a look at singular vectors
plt.figure(10)
plt.clf()
plt.plot(V[:, 29], 'k')
plt.show()