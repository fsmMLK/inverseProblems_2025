import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipySignal
from scipy import linalg as scipyLinalg

def plotData(f_true, f_recovered, title,labelRec='Recovered f'):
    # Compare recovered f with original
    plt.figure(figsize=(8, 4))
    plt.plot(f_true, label='Original f')
    plt.plot(f_recovered, label=labelRec, linestyle='--')
    plt.legend()
    plt.title(title)
    plt.show()

# Step 1: Define the signal f in R^100
n=100
f = np.zeros(n)
f[24:75 + 1] = 1.0  # 25 to 75 inclusive (Python index starts at 0)

# Step 2: Construct the convolution matrix A
std=2
psf = scipySignal.windows.gaussian(11,std)
psf = psf / np.sum(psf)  # normalize
A = scipyLinalg.convolution_matrix(psf, n, mode='same')

# Step 3: Construct the convolution matrix A2
std=2.1
psf = scipySignal.windows.gaussian(11,std)
psf = psf / np.sum(psf)  # normalize
A2 = scipyLinalg.convolution_matrix(psf, n, mode='same')

# Step 4: Generate synthetic datum m = A f and recover
m = A @ f
delta = 1e-4
eps = np.random.normal(0, delta, size=n)
m_delta = m + eps

plt.figure(figsize=(8, 4))
plt.plot(f, label='Original f')
plt.plot(m, label='Measurement m', linestyle='--')
plt.legend()
plt.title('Original f and Measurement m')
plt.show()

#Step 5: Recover f using A⁻¹ (inverse crime, no noise)
f_1 = scipyLinalg.solve(A, m)
plotData(f, f_1, "no noise / with inverse crime")
print("error norm f_1: ",scipyLinalg.norm(f - f_1))

#Step 6: Recover f using A⁻¹ (no inverse crime, no noise)
f_2 = scipyLinalg.solve(A2, m)
plotData(f, f_2, "no noise / no inverse crime")
print("error norm f_2: ",scipyLinalg.norm(f - f_2))

#Step 7: Recover f using A⁻¹ (inverse crime, with noise)
f_3 = scipyLinalg.solve(A, m_delta)
plotData(f, f_3, "with noise / with inverse crime")
print("error norm f_3: ",scipyLinalg.norm(f - f_3))

#Step 6: Recover f using A⁻¹ (no inverse crime, no noise)
f_4 = scipyLinalg.solve(A2, m_delta)
plotData(f, f_4, "with noise / no inverse crime")
print("error norm f_4: ",scipyLinalg.norm(f - f_4))

