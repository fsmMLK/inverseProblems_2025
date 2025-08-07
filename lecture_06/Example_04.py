import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg


# 1 Create the signal
n = 100
x = np.arange(1, n+1).reshape(-1, 1)
n0 = int(np.floor(n/15))
n1, n2, n3, n4, n5, n6 = 2*n0, 3*n0, 4*n0, 8*n0, 10*n0, 14*n0
f = ((x > n1) & (x <= n2)).astype(float) + 3*(x-n3)/(n4-n3)*((x > n3) & (x <= n4)) - (1 + np.cos(x*2*np.pi/(n6-n5)))*((x > n5) & (x <= n6))

# Point spread function from a continuous function
PSF = np.array([1, 1, 1])
PSF = np.convolve(PSF, PSF)
PSF = np.convolve(PSF, PSF)
PSF = np.convolve(PSF, PSF)
PSF = PSF / np.sum(PSF)

# Create convolution matrix
A = convmtx(PSF, n)
nu = (len(PSF)-1) // 2
A = A[:, nu:nu+n]

# Generate data
m = A @ f
delta = 1e-3
noise = np.random.randn(n, 1)
noise = noise / np.linalg.norm(noise)
md = m + delta * noise

# 2 create the sample test for alpha
ALPHA = np.logspace(-9, -3, 50)

# 3 If I knew the solution...
error = np.zeros_like(ALPHA)
for i in range(len(ALPHA)):
    alpha = ALPHA[i]
    fd = np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ md)
    error[i] = np.linalg.norm(fd - f) / np.linalg.norm(f)

plt.figure(1)
plt.loglog(ALPHA, error, 'bo-')
i = np.argmin(error)
plt.loglog(ALPHA[i], error[i], 'r.', markersize=25)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$|| T_\alpha m_\delta - f ||$')
plt.title(f'Real best one: {ALPHA[i]}')

# 4 Morozov
res = np.zeros_like(ALPHA)
for i in range(len(ALPHA)):
    alpha = ALPHA[i]
    fd = np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ md)
    res[i] = np.linalg.norm(A @ fd - md)

plt.figure(2)
plt.loglog(ALPHA, res, 'bo-')
i = np.argmin(np.abs(res - delta))
plt.title(f'Morozov: {ALPHA[i]}')
plt.loglog(ALPHA, delta * np.ones_like(ALPHA), '--k')
plt.loglog(ALPHA[i], res[i], 'r.', markersize=25)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$r(\alpha)$')

# 5 L curve
X = np.zeros_like(ALPHA)
Y = np.zeros_like(ALPHA)
for i in range(len(ALPHA)):
    alpha = ALPHA[i]
    fd = np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ md)
    X[i] = np.log(np.linalg.norm(A @ fd - md))
    Y[i] = np.log(np.linalg.norm(fd))

plt.figure(3)
plt.plot(X, Y, 'bo-')

# Rescale X and Y
Xn = (X - np.min(X)) / (np.max(X) - np.min(X))
Yn = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
i = np.argmin(Xn**2 + Yn**2)  # find the closest point to the origin

plt.title(f'L curve: {ALPHA[i]}')
plt.plot(X[i], Y[i], 'r.', markersize=25)
plt.xlabel('log(r($\\alpha$))')
plt.ylabel('log(|| T_$\\alpha$ m_$\\delta$ ||)')

plt.show()