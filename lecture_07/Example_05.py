import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg
from Example_02 import generateSignal, generatePSF,generateMeasurements

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
plt.ylabel(r'$|| T_\alpha m_\delta - f ||$')
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
plt.ylabel(r'$r(\alpha)$')
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
plt.xlabel('log(r($\\alpha$))')
plt.ylabel('log(|| T_$\\alpha$ m_$\\delta$ ||)')
plt.grid(True)
plt.show()