import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipySignal
from scipy import linalg as scipyLinalg

"""
#Example 02 - Convolution

This example demonstrates the process of 1D signal convolution using Python. It starts by defining a simple rectangular signal and a normalized
 point spread function (PSF). The example shows how to perform convolution using both the built-in scipy.signal.convolve function and a
 manually built convolution matrix approach.

 Fernando Moura, 2025
"""

"""## Step 1: Define the signal $f\in R^{100}$"""
 
n=100
f = np.zeros(n)
f[24:75 + 1] = 1.0  # 25 to 75 inclusive (Python index starts at 0)

# Plot signal f
plt.figure(figsize=(10, 4))
plt.stem(f)
plt.title('Original Signal $f$')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

"""## Step 2: Define the point spread function $p$."""

p = np.array([1, 2, 3, 2, 1], dtype=float) # Ensure p has odd length
p /= p.sum()  # Normalize

"""#Step 3: Convolution with zero padding using **scipy.signal.convolve**"""

# mode='same' : returns the central part of the convolution that is the same size as f.
#               zero padding is handled by default in 'convolve' with 'same' mode
# mode='valid': would return a smaller array, not including the edges to avoid zero padding
# mode='full' : would return the full convolution result
conv_result = scipySignal.convolve(f, p, mode='same')

plt.figure(figsize=(10, 4))
plt.stem(conv_result,)
plt.title('Convolution using scipy.signal.convolve')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

"""## Step 4: Convolution using convolution matrix (**scipy.linalg.convolution_matrix**)"""

convMtx = scipyLinalg.convolution_matrix(p, len(f), mode='same')
plt.figure(figsize=(10, 4))
plt.spy(convMtx, markersize=1)
plt.title('convMtx sparsity pattern')
plt.show()

conv_result_matrix = convMtx @ f

plt.figure(figsize=(10, 4))
plt.stem(conv_result_matrix)
plt.title('Convolution using scipy.linalg.convolution_matrix')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

"""## Step 5: Compare original and convolved signals"""

plt.figure(figsize=(12, 6))
plt.plot(f, label='Original Signal $f$')
plt.plot(conv_result, label='Convolve Result (same length)')
plt.plot(conv_result_matrix, '--', label='Convmtx Result (same length)')
plt.title('Comparison of Convolution Results')
plt.legend()
plt.grid(True)
plt.show()

"""## Step 6: Manually build convolution matrix with zero padding"""

def manual_convmtx(p, ncols):

    if len(p) % 2 == 0:
        raise ValueError("Point spread function p must have an odd length.")

    n = ncols
    conv_matrix = (p[4] * np.diagflat(np.ones(n - 2), -2) +
                   p[3] * np.diagflat(np.ones(n - 1), -1) +
                   p[2] * np.diagflat(np.ones(n), 0) +
                   p[1] * np.diagflat(np.ones(n - 1), 1) +
                   p[0] * np.diagflat(np.ones(n - 2), 2))
    return conv_matrix

convMtxManual = manual_convmtx(p, len(f))

# Verify it matches the previous result
manual_conv_result = convMtxManual @ f
assert np.allclose(manual_conv_result, conv_result_matrix)
print('Manually built conv Matrix is equal to Built-in function')

# Visualize sparsity structure
plt.figure(figsize=(6, 6))
plt.spy(convMtxManual, markersize=1)
plt.title('Manually created convolution matrix sparsity pattern')
plt.show()
