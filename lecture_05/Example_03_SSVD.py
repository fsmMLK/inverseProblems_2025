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

def calcPicardPlot(A,m,axis, title='Discrete Picard Plot', flagShow=False):
    U, S, Vt = scipyLinalg.svd(A)

    UTb = np.abs(U.T @ m)  # |u_i^T m|
    picard_coeffs = UTb / S  # |u_i^T m| / sigma_i

    axis.semilogy(S, 'or', label='Singular values (σ_i)')
    axis.semilogy(UTb, 'xg', label='|u_i^T m|')
    axis.semilogy(picard_coeffs, 'db', label='|u_i^T m| / σ_i')
    axis.set_title(title)
    axis.set_ylabel('Magnitude (log scale)')
    axis.set_xlabel('Index i')
    axis.legend()
    axis.grid(True)

    if flagShow:
        plt.show()

"""# Step 1: Define the signal $f\in R^{100}$"""
f = np.zeros(100)
f[24:76] = 1  # 25 to 75 inclusive (Python index starts at 0)


# Plot the signal f
plt.figure(figsize=(10, 4))
plt.stem(f)
plt.title('Original Signal $f$')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

"""# Step 2: Define the point spread function p."""

p = np.array([1, 2, 3, 2, 1], dtype=float) #  ensure p has odd length
p /= p.sum()  # Normalize

"""# Step 3: Convolution using convolution matrix (convmtx)"""

A = scipyLinalg.convolution_matrix(p, len(f), mode='same')

"""# Steps 4: generate datum"""

std1=1e-2
m = A @ f  # Convolve f with p
m_d1 = m + std1 * np.random.randn(*m.shape)

"""# Step 5: Naive inversion"""

f_naive_d1 = scipyLinalg.solve(A, m_d1)  # Naive inversion with small noise

"""# Step 6: Create the picard plot"""

fig, ax = plt.subplots(1, 1, figsize=(15, 5),sharey=True)
calcPicardPlot(A,m_d1,ax,flagShow=True)

"""# Step 7: Compute TSVD solution"""

U, svals, Vt = scipyLinalg.svd(A)

r_alpha = 40
svals_inv_ralpha = np.zeros_like(svals)
svals_inv_ralpha[:r_alpha] = 1.0/svals[:r_alpha]

f_tsvd = Vt.T @ np.diag(svals_inv_ralpha) @ U.T @ m_d1

"""# Step 8: Compute SSVD solution"""

tau=1e-2
UTb = np.abs(U.T @ m_d1)
svals_inv=np.zeros_like(svals)
for i,s in enumerate(svals):
    if UTb[i]<tau:
        svals_inv[i]=0# |u_i^T m|
    else:
        svals_inv[i]=1/svals[i]
svals_inv[40:]=0

f_ssvd= Vt.T @ np.diag(svals_inv) @ U.T @ m_d1  # SSVD solution

# Plot the original and naive inversion results
plt.figure(figsize=(10, 4))
plt.plot(f, label='Original signal')
#plt.plot(f_naive_d1, label='Naive Inversion')
plt.plot(f_ssvd, label='SSVD')
plt.plot(f_tsvd, label='TSVD')
plt.title('Naive Inversion Result (std=%1.1e)' % std1 )
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()