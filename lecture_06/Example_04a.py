import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg
from Example_01 import generateSignal
from Example_02 import generatePSF,generateMeasurements

"""# Step 1: Create signal, PSF and data"""
n = 200
x, f = generateSignal(n,showPlot=False)

A,PSF = generatePSF(n)

delta = 1e-2
m,md = generateMeasurements(A,f,delta)

"""# Step 2: regularization operator and vector"""
n0 = round(n / 15)
n1 = 2 * n0
n2 = 3 * n0
n3 = 4 * n0
n4 = 8 * n0

if True:
    LA=np.zeros((n1,n1))
    LB=10*np.eye(n2-n1)
    LC=np.zeros((n3-n2,n3-n2))
    LD=1*np.eye(n4-n3)
    LE=np.zeros((n-n4,n-n4))
else:
    LA=np.eye(n1)
    LB=10*np.eye(n2-n1)
    LC=np.eye(n3-n2)
    LD=1*np.eye(n4-n3)
    LE=np.eye(n-n4)


L = scipyLinalg.block_diag(LA,LB,LC,LD,LE)
f_star = np.zeros_like(f)
f_star[n1:n2] = 1
f_star[n3:n4] = 1.5

"""# Step 3: solve the generalized Tikhonov problem"""
alpha = 0.1
fd = np.linalg.solve(A.T @ A + alpha * (L.T @ L), A.T @ md + alpha * L.T @ L @ f_star)

plt.figure()
plt.plot(x, f, 'k', label='Original signal')
plt.plot(x, md, 'r', label='noisy data')
plt.plot(x, fd, 'b', label='Generalized Tikhonov')
plt.legend()
plt.axis([0, n, -3, 4])
plt.grid(True)
plt.show()





