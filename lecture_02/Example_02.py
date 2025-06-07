import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipySignal
from scipy import linalg as scipyLinalg

# Step 1: Define the signal f in R^100
n=100
f = np.zeros(n)
f[24:75 + 1] = 1.0  # 25 to 75 inclusive (Python index starts at 0)

# Point spread function p, convolve with itself and normalize
p = np.array([1, 1, 1, 1, 1], dtype=float)
psf = scipySignal.convolve(p, p, mode='full')
psf = psf / np.sum(psf)  # normalize

# Step 2: Construct the convolution matrix A
A = scipyLinalg.convolution_matrix(psf, n, mode='same')

# Step 3: Generate synthetic datum m = A f and recover f using A⁻¹
m = A @ f
f_recovered = scipyLinalg.solve(A, m)

# Compare recovered f with original
plt.figure(figsize=(10, 4))
plt.plot(f, label='Original f')
plt.plot(f_recovered, label='Recovered f (no noise)', linestyle='--')
plt.legend()
plt.title("Reconstruction without noise")
plt.show()

error_no_noise = scipyLinalg.norm(f - f_recovered)
print(f"Error without noise: {error_no_noise:.2e}")

# Step 4: Add Gaussian noise with delta = 1e-4
delta = 1e-4
eps = np.random.normal(0, delta, size=n)
m_delta = m + eps
f_delta_rec = scipyLinalg.solve(A, m_delta)

plt.figure(figsize=(10, 4))
plt.plot(f, label='Original f')
plt.plot(f_delta_rec, label=f'Recovered f (delta={delta})', linestyle='--')
plt.legend()
plt.title(f"Reconstruction with noise (delta = {delta})")
plt.show()

error_noise_1 = scipyLinalg.norm(f - f_delta_rec)
print(f"Error with noise (delta = {delta}): {error_noise_1:.2e}")

# Step 5: Repeat with delta = 0.01
delta = 0.01
eps = np.random.normal(0, delta, size=n)
m_delta = m + eps
f_delta_rec = scipyLinalg.solve(A, m_delta)

plt.figure(figsize=(10, 4))
plt.plot(f, label='Original f')
plt.plot(f_delta_rec, label=f'Recovered f (delta={delta})', linestyle='--')
plt.legend()
plt.title(f"Reconstruction with noise (delta = {delta})")
plt.show()

error_noise_2 = scipyLinalg.norm(f - f_delta_rec)
print(f"Error with noise (delta = {delta}): {error_noise_2:.2e}")

# Step 6: Use a different PSF (p convolved with itself three times)
p = np.array([1, 1, 1, 1, 1], dtype=float)
p = scipySignal.convolve(p, p, mode='full')
p = scipySignal.convolve(p, p, mode='full')
p = scipySignal.convolve(p, p, mode='full')
psf3 = p / np.sum(p)

# Construct new A matrix
A2 = scipyLinalg.convolution_matrix(psf3, n, mode='same')

# Create new m and solve for f
m3 = A2 @ f
delta = 0.01
eps = np.random.normal(0, delta, size=n)
m3_delta = m3 + eps
f3_rec = scipyLinalg.solve(A2, m3_delta)

plt.figure(figsize=(10, 4))
plt.plot(f, label='Original f')
plt.plot(f3_rec, label='Recovered f (PSF³, delta=0.01)', linestyle='--')
plt.legend()
plt.title("Reconstruction with 3x convolved PSF and noise")
plt.show()

error_psf3 = scipyLinalg.norm(f - f3_rec)
print(f"Error with 3x convolved PSF and delta=0.01: {error_psf3:.2e}")
