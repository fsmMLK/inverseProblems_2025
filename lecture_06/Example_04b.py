import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg
from Example_01 import generateSignal
from Example_02 import generatePSF,generateMeasurements

"""# Step 1: Create signal, PSF and data"""
n = 200
x, f = generateSignal(n,showPlot=False)
f[8 *(round(n / 15)):] += 1

A,PSF = generatePSF(n)

delta = 1e-2
m,md = generateMeasurements(A,f,delta)

"""# Step 2: regularization operator and vector"""
c=0.01
n0=50
idx=np.arange(n0)
LA=np.diag(np.exp(-c*idx))
LB=np.diag(np.exp(c*np.flip(idx)))
LC=np.zeros((n-2*n0,n-2*n0))
L = scipyLinalg.block_diag(LA,LC,LB)
f_star = np.zeros_like(f)
f_star[(n-n0):] = 1.

"""# Step 3: solve the generalized Tikhonov problem"""
alpha = 0.001
fd = np.linalg.solve(A.T @ A + alpha * (L.T @ L), A.T @ md + alpha * L.T @ L @ f_star)

plt.figure()
plt.plot(x, f, 'k', label='Original signal')
plt.plot(x, md, 'r', label='noisy data')
plt.plot(x, fd, 'b', label='Generalized Tikhonov')
plt.legend()
plt.axis([0, n, -3, 4])
plt.grid(True)
plt.show()





