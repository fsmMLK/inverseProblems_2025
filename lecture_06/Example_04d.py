import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg
from Example_04a import generateSignal
from Example_02 import generatePSF,generateMeasurements
from Example_03 import generateDiffOperator

def generateSignal(n, polyOrder=5, showPlot=True):
    """
    Generate a synthetic signal for testing.
    """
    roots=np.random.rand(polyOrder)
    p = np.flip(np.polynomial.polynomial.polyfromroots(roots))

    x = np.linspace(0, n, n)/n
    f= np.polyval(p,x)
    if showPlot:
        plt.plot(x, f, label='Original Signal')
        plt.xlabel('Index')
        plt.ylabel('Signal Value')
        plt.title('Generated Signal')
        plt.legend()
        plt.grid(True)
        plt.show()
    return x, f


"""# Step 1: Create signal, PSF and data"""
n = 200
polyOrder=4
x, f = generateSignal(n,polyOrder,showPlot=False)

A,PSF = generatePSF(n)

delta = 1e-1
m,md = generateMeasurements(A,f,delta)

"""# Step 2: regularization operator and vector"""
# build polinomial base
priorBasisOrder=polyOrder
B=np.zeros((n,priorBasisOrder+1))
for i in range(priorBasisOrder+1):
    poly=[1,]*(i+1)
    #print(poly)
    B[:,i]=np.polyval(poly, x)

plt.figure()
plt.plot(x, B, 'k')
plt.grid(True)
plt.title('Solution basis')
plt.show()

# projection matrix PB
PB=B@scipyLinalg.inv(B.T@B)@B.T

L=np.eye(n)-PB
f_star = np.zeros_like(f)

"""# Step 3: solve the generalized Tikhonov"""
alpha = 10
fd = np.linalg.solve(A.T @ A + alpha * (L.T @ L), A.T @ md + alpha * L.T @ L @ f_star)

plt.figure()
plt.plot(x, f, 'k', label='Original signal')
plt.plot(x, md, 'r', label='noisy data')
plt.plot(x, fd, 'b', label='Generalized Tikhonov')
plt.legend()
#plt.axis([0, n, -3, 4])
plt.grid(True)
plt.show()





