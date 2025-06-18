import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scipyLinalg

# Step 1: Define the signal f in R^100
f = np.zeros(100)
f[24:75] = np.linspace(0,50,51)/50  # 25 to 75 inclusive (Python index starts at 0)

# Plot the signal f
plt.figure(figsize=(10, 4))
plt.stem(f)
plt.title('Original Signal $f$')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Step 2: Define the point spread function p. ensure p has odd length
p = np.array([1, 2, 3, 2, 1], dtype=float)
p /= p.sum()  # Normalize

# Step 3: Convolution using convolution matrix (convmtx)
A = scipyLinalg.convolution_matrix(p, len(f), mode='same')

# Step 4 and 5: Naive inversion.
m = A @ f  # Convolve f with p
f_naive = scipyLinalg.solve(A, m)  # Naive inversion, no noise

# add noise to the convolved signal
std1=1e-4
m_d1=m+std1*np.random.randn(*m.shape)
std2=1e-3
m_d2=m+std2*np.random.randn(*m.shape)

f_naive_d1 = scipyLinalg.solve(A, m_d1)  # Naive inversion with small noise
f_naive_d2 = scipyLinalg.solve(A, m_d2)  # Naive inversion with moderate noise

# Plot the original and naive inversion results
plt.figure(figsize=(10, 4))
plt.plot(f, label='Original signal')
plt.plot(f_naive, label='Naive Inversion')
plt.plot(f_naive_d1, label='Naive Inversion (std=%1.1e)'%std1)
plt.plot(f_naive_d2, label='Naive Inversion (std=%1.1e)'%std2)
plt.title('Naive Inversion Result')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Compute A^{-1}
A_inv = scipyLinalg.inv(A)

# Step 7: Compute norm of A^{-1}
print(f"Norm of A_inv: %f" % scipyLinalg.norm(A_inv, 2))
U, S, Vt = scipyLinalg.svd(A)

# Plot singular values
plt.figure(figsize=(10, 4))
plt.semilogy(S, 'o-')
plt.title('Singular Values of A')
plt.xlabel('Index')
plt.ylabel('Singular Value (log scale)')
plt.grid(True)
plt.show()

# Condition number
print(f"Condition number of A: %f"%(S[0] / S[-1]))

# Step 8: Compute norm of A^{-1}

# Compute Picard coefficients |u_i^T b| / sigma_i for the case of no noise
UTb = np.abs(U.T @ m)  # |u_i^T b|
picard_coeffs = UTb / S  # |u_i^T b| / sigma_i

# Discrete Picard plot
plt.figure(figsize=(10, 5))
plt.semilogy(S, 'or', label='Singular values (σ_i)')
plt.semilogy(UTb, 'xg', label='|u_i^T b|')
plt.semilogy(picard_coeffs, 'db', label='Picard coefficients (|u_i^T b| / σ_i)')
plt.title('Discrete Picard Plot')
plt.xlabel('Index i')
plt.ylabel('Magnitude (log scale)')
plt.legend()
plt.grid(True)
plt.show()
