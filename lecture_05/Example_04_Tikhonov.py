import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as scipyLinalg

saveFig = False

delta = 1e-5   # noise level
alpha = 1e-1   # regularization parameter

"""Step 1: Create the signal"""

n = 200
x = np.linspace(0, n, n)
f = np.zeros(n)
n0 = round(n / 15)
n1 = 2 * n0
n2 = 3 * n0
n3 = 4 * n0
n4 = 8 * n0
n5 = 10 * n0
n6 = 14 * n0

f[n1:n2] = 1
f[n3:n4] = 3 * (np.arange(n3, n4) - n3) / (n4 - n3)
f[n5:n6] = -1 - np.cos(np.arange(n5, n6) * 2 * np.pi / (n6 - n5))

plt.figure()
plt.plot(x, f, 'k', linewidth=2)
plt.axis([0, n, -2, 3])
plt.title('Original signal')
plt.grid(True)

if saveFig:
    plt.savefig('Ex04_image_signal.svg', format='svg', bbox_inches='tight')
plt.show()

"""Step 2: Point Spread Function and convolution matrix"""

psf = np.array([1, 4, 8, 16, 19, 15, 10, 7, 1])
psf = psf / np.sum(psf)
plt.figure()
plt.plot(psf, 'ro-')
plt.title('Point spread function')
plt.show()

# convolution matrix
A = scipyLinalg.convolution_matrix(psf, n, mode='same')

"""# Step 3: Noisy measurements"""

m = A @ f
r_noise = np.random.randn(len(m))
r_noise = r_noise / np.linalg.norm(r_noise)
md = m + delta * r_noise

plt.figure()
plt.plot(x, f, 'k', label='Original signal')
plt.plot(x, md, 'r', label='noisy data')
plt.legend()
plt.axis([0, n, -2, 3])
plt.grid(True)
plt.title('Noisy data')
if saveFig:
    plt.savefig('Ex04_image_noisy_data.svg', format='svg', bbox_inches='tight')
plt.show()

"""# Step 4: Pseudoinverse"""

A_pinv = scipyLinalg.pinv(A)
f_d = A_pinv @ md

plt.figure()
plt.plot(x, f, 'k', label='Original signal')
plt.plot(x, f_d, 'g', label='Pseudo-inverse solution')
plt.legend()
plt.axis([0, n, -2, 3])
plt.grid(True)
plt.title('Noisy data')
if saveFig:
    plt.savefig('Ex04_image_noisy_data.svg', format='svg', bbox_inches='tight')
plt.show()

"""# Step 5: Tikhonov regularization"""

U, svals, Vt = scipyLinalg.svd(A)
svals_tik = np.diag(svals / (svals ** 2 + alpha))
T_a = Vt.T @ svals_tik @ U.T
f_ad = T_a @ md

"""# Step 6: Comparison"""

error_PS = np.linalg.norm(f_d - f) / np.linalg.norm(f)
error_Tik = np.linalg.norm(f_ad - f) / np.linalg.norm(f)
print(f' Error with pseudo-inverse: \t {error_PS:.3f}')
print(f' Error with Tikhonov regul: \t {error_Tik:.3f}')

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x, f_ad, 'g')
plt.plot(x, f, 'b')
plt.axis([0, n, -2, 3])
plt.title('Tikhonov error: %1.2g' % error_Tik)

plt.subplot(1, 2, 2)
plt.plot(x, f_d, 'k')
plt.plot(x, f, 'b')
plt.axis([0, n, -2, 3])
plt.title('Pseudoinverse error: %1.2g' % error_PS)

plt.show()

"""# Step 7: change alpha and delta"""
