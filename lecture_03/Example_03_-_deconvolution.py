import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scipyLinalg

"""
This example explores 1D signal convolution and deconvolution using Python. It begins by constructing a simple rectangular signal and a normalized 
point spread function (PSF). The convolution is performed both via built-in functions and noise is added to the resulting data. The example then 
addresses the inverse problem, demonstrating naive inversion and highlighting its sensitivity to noise. Singular value decomposition (SVD) and 
Picard plots are used to analyze the stability and conditioning of the inversion.

Fernando Moura, 2025
"""

"""## Step 1: Define the signal $f\in R^{100}$"""

f = np.zeros(100)
f[24:75] = np.linspace(0, 50, 51) / 50  # 25 to 75 inclusive (Python index starts at 0)

# Plot the signal f
plt.figure(figsize=(10, 4))
plt.stem(f)
plt.title('Original Signal $f$')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

"""## Step 2: Define the point spread function p."""

p = np.array([1, 2, 3, 2, 1], dtype=float) #  ensure p has odd length
p /= p.sum()  # Normalize

"""## Step 3: Convolution using convolution matrix (convmtx)"""

A = scipyLinalg.convolution_matrix(p, len(f), mode='same')

"""## Steps 4, 5, and 6: Naive inversion."""

m = A @ f  # Convolve f with p
f_naive = scipyLinalg.solve(A, m)  # Naive inversion, no noise

# add noise to the convolved signal
std1 = 1e-4
m_d1 = m + std1 * np.random.randn(*m.shape)
std2 = 1e-3
m_d2 = m + std2 * np.random.randn(*m.shape)

f_naive_d1 = scipyLinalg.solve(A, m_d1)  # Naive inversion with small noise
f_naive_d2 = scipyLinalg.solve(A, m_d2)  # Naive inversion with moderate noise

# Plot the original and naive inversion results
plt.figure(figsize=(10, 4))
plt.plot(f, label='Original signal')
plt.plot(f_naive, label='Naive Inversion')
plt.plot(f_naive_d1, label='Naive Inversion (std=%1.1e)' % std1)
plt.plot(f_naive_d2, label='Naive Inversion (std=%1.1e)' % std2)
plt.title('Naive Inversion Result')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

"""## Step 7: Compute SVD of A"""

U, S, Vt = scipyLinalg.svd(A)

# Plot singular values
plt.figure(figsize=(10, 4))
plt.semilogy(S, 'o-')
plt.title('Singular Values of A')
plt.xlabel('Index')
plt.ylabel('Singular Value (log scale)')
plt.grid(True)
plt.show()

"""## Step 8: Create a discrete Picard plot for $m$, $m_{\delta1}$, and  $m_{\delta2}$"""

UTb = np.abs(U.T @ m)  # |u_i^T m|
picard_coeffs = UTb / S  # |u_i^T m| / sigma_i

UTb_d1 = np.abs(U.T @ m_d1)  # |u_i^T m|
picard_coeffs_d1 = UTb_d1 / S  # |u_i^T m| / sigma_i

UTb_d2 = np.abs(U.T @ m_d2)  # |u_i^T m|
picard_coeffs_d2 = UTb_d2 / S  # |u_i^T m| / sigma_i

# Discrete Picard plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5),sharey=True)

axes[0].semilogy(UTb, 'xg', label='|u_i^T m|')
axes[0].semilogy(picard_coeffs, 'db', label='Picard coefficients (|u_i^T m| / σ_i)')
axes[0].set_title('Discrete Picard Plot (no noise)')
axes[0].set_ylabel('Magnitude (log scale)')

axes[1].semilogy(UTb_d1, 'xg', label='|u_i^T m|')
axes[1].semilogy(picard_coeffs_d1, 'db', label='Picard coefficients (|u_i^T m| / σ_i)')
axes[1].set_title('Discrete Picard Plot (noise 1e-4)')

axes[2].semilogy(UTb_d2, 'xg', label='|u_i^T m|')
axes[2].semilogy(picard_coeffs_d2, 'db', label='Picard coefficients (|u_i^T m| / σ_i)')
axes[2].set_title('Discrete Picard Plot (noise 1e-3)')

for ax in axes:
    ax.semilogy(S, 'or', label='Singular values (σ_i)')
    ax.set_xlabel('Index i')
    ax.legend()
    ax.grid(True)

plt.show()
