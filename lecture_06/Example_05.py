import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scipyLinalg
from scipy.linalg import toeplitz, block_diag
from Example_01 import generateSignal
from Example_02 import generateMeasurements, generatePSF
from Example_03 import generateDiffOperator

def solveLinearSysTikhonov(A, m, L, f0, alpha):
    """
    Solve linear system using Tikhonov regularization
    using normal equations formulation

    f_alpha = argmin_{f}  || Af - m ||^2 + alpha*|| L(f - f0) ||^2

            <---- A_tilde --->        <---- m_tilde --->
    f_alpha=(A'*A + alpha*L'*L)^{-1}*(A'*m+alpha*L'*L*f0)
    f_alpha= A_tilde^{-1}*m_tilde

    Parameters:
    -----------
    A : numpy array (MxN)
    m : numpy array (M,) - measurement vector
    L : numpy array (PxN) - regularization matrix
    f0 : numpy array (N,) - prior solution
    alpha : float - regularization parameter

    Returns:
    --------
    f_alpha : numpy array (N,) - regularized solution
    """
    A_tilde = A.T @ A + alpha * L.T @ L
    m_tilde = A.T @ m + alpha * L.T @ L @ f0

    # Solve using numpy's linear algebra solver (equivalent to MATLAB's backslash)
    f_alpha = np.linalg.solve(A_tilde, m_tilde)

    return f_alpha

def solveLinearSysTikhonov2(A, m, L, f0, alpha, M, g0, beta):
    """
    Solve linear system using Tikhonov regularization with two regularization terms

    f_(alpha,beta) = argmin_{f}  || Af - m ||^2 + alpha*|| L(f - f0) ||^2 + beta*|| M(f - g0) ||^2

            <--------- A_tilde --------->        <---- m_tilde --->
    f_alpha=(A'*A + alpha*L'*L+ beta*M'*M)^{-1}*(A'*m+alpha*L'*L*f0+beta*M'*M*g0)
    f_alpha= A_tilde^{-1}*m_tilde

    Parameters:
    -----------
    A : numpy array (MxN)
    m : numpy array (M,) - measurement vector
    L : numpy array (PxN) - first regularization matrix
    f0 : numpy array (N,) - first prior solution
    alpha : float - first regularization parameter
    M : numpy array (QxN) - second regularization matrix
    g0 : numpy array (N,) - second prior solution
    beta : float - second regularization parameter

    Returns:
    --------
    f_alpha : numpy array (N,) - regularized solution
    """
    A_tilde = A.T @ A + alpha * L.T @ L + beta * M.T @ M
    m_tilde = A.T @ m + alpha * L.T @ L @ f0 + beta * M.T @ M @ g0

    # Solve using numpy's linear algebra solver
    f_alpha = np.linalg.solve(A_tilde, m_tilde)

    return f_alpha

def plotSolution(figureNbr, x, f_true, md, f_alpha, plotTitle=None):
    """
    Plot linear system solution

    Parameters:
    -----------
    figureNbr : int - figure number
    x : numpy array - x-axis values
    f_true : numpy array - original signal
    md : numpy array - convolved signal + noise
    f_alpha : numpy array - reconstructed signal
    plotTitle : str - plot title
    """
    plt.figure(figureNbr)
    plt.plot(x, f_true, 'k', label='original signal', linewidth=2)
    plt.plot(x, md, 'r', label='convolved signal + noise', linewidth=2)
    plt.plot(x, f_alpha, 'b', label='reconstructed signal', linewidth=2)
    if plotTitle is not None:
        plt.title(plotTitle)
    plt.ylim([-3, 3])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# plot opts
linewidth = 2

"""# Step 1: Create signal, PSF and data"""
n = 200
x, f = generateSignal(n, showPlot=False)

"""# 2 Point Spread Function and convolution matrix"""
A,PSF = generatePSF(n)

delta = 1e-3
m, md = generateMeasurements(A, f, delta)

plt.figure(1)
plt.plot(x, f, 'k', linewidth=linewidth)
plt.plot(x, md, 'r', linewidth=linewidth)
plt.title('Noisy data')
plt.legend(['original signal', 'convolved signal + noise'])
plt.show()

# -------------------------------------------
# Tikhonov regularization L=weight factor, f0=averages
# -------------------------------------------
f0 = np.zeros(n)
n0 = round(n / 15)
n1 = 2 * n0
n2 = 3 * n0
n3 = 4 * n0
n4 = 8 * n0
f0[n1:n2] = 1  # with high confidence
f0[n3:n4] = 1.5  # we know the value 'in average' in the interval

L = np.zeros((n, n))
L[n1:n2, n1:n2] = 10 * np.eye(n2 - n1)  # with high confidence
L[n3:n4, n3:n4] = 1 * np.eye(n4 - n3)  # with lower confidence

figureNbr = 2
alpha_1 = 1e-1
f_alpha = solveLinearSysTikhonov(A, md, L, f0, alpha_1)
plotSolution(figureNbr + 1, x, f, md, f_alpha, 'Regularized solution')

# -------------------------------------------
# Tikhonov: add another term to the regularization M=I, g0=0
# -------------------------------------------
M = np.eye(n)
g0 = np.zeros(n)
figureNbr = 3

alpha_2 = 1e-4
f_alpha = solveLinearSysTikhonov2(A, md, L, f0, alpha_1, M, g0, alpha_2)
plotSolution(figureNbr + 1, x, f, md, f_alpha, f'Regularized solution')
