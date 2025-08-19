import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scipyLinalg
from scipy.linalg import toeplitz, block_diag
from Example_01 import generateSignal, generateMeasurements, generatePSF


def generateDiffOperator(size):
    """
    Generate a finite difference operator for 1D signals.
    """
    D = np.zeros((size - 1, size))
    for i in range(size - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D


def calcMorozov(A, md, L, f0, alphaMin, alphaMax, nAlphas, normDelta, flagPlotResiduals, figureNbr):
    """
    Computes (using brute force) the best alpha parameter in an interval
    obs: this function tries a range of values and chooses the one with the
    smallest residual. This is not the theoretical optimal value!

    residual:   r= || A f_alpha - md ||

    where f_alpha is the solution of the regularized problem
    f_alpha = argmin_{f}  || Af - md ||^2 + alpha*|| L(f - f0) ||^2

    Parameters:
    -----------
    A : numpy array (MxN)
    md : numpy array (M,) - noisy measurement vector
    L, f0: tikhonov regularization matrix (PxN) and vector (N,)
    alphaMin (int): min alpha value, in terms of powers of 10. Ex: for 1e-5 use alphaMin=-5
    alphaMax (int): max alpha value, in terms of powers of 10. Ex: for 1e2 use alphaMax=2
    nAlphas (int): number of alpha values in the interval (log scale!)
    normDelta (float): norm of the error delta_m: remember    md = m + delta_m
    flagPlotResiduals (boolean): create a plot or not
    figureNbr: number of the figure. used only if flagPlotResiduals=True

    Returns:
    --------
    bestAlpha : float - best alpha value
    residuals : numpy array - residuals for each alpha value

    Fernando Moura Oct 2021
    """

    alphaVals = np.logspace(alphaMin, alphaMax, nAlphas)

    residuals = np.zeros(nAlphas)
    for i in range(nAlphas):
        fda = solveLinearSysTikhonov(A, md, L, f0, alphaVals[i])
        residuals[i] = np.linalg.norm(A @ fda - md)

    # find min residual
    bestIdx = np.argmin(np.abs(residuals - normDelta))
    bestAlpha = alphaVals[bestIdx]

    if flagPlotResiduals:
        plt.figure(figureNbr)
        plt.loglog(alphaVals, residuals, 'bo-')
        plt.title(f'Morozov best alpha={bestAlpha}')
        plt.loglog(alphaVals, normDelta * np.ones_like(alphaVals), '--k')
        plt.loglog(bestAlpha, residuals[bestIdx], 'r.', markersize=15)
        plt.text(alphaVals[1], normDelta * 1.1, r'||$\delta$||', fontsize=15, verticalalignment='bottom')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$r(\alpha)$')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.show()

    return bestAlpha, residuals


def plotSolution(figureNbr, x, f_true, md, f_alpha, plotTitle):
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
    plt.title(plotTitle)
    plt.ylim([-3, 3])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


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


# plot opts
linewidth = 2

"""# Step 1: Create signal, PSF and data"""
N = 200
x, f = generateSignal(N, showPlot=False)

"""# 2 Point Spread Function and convolution matrix"""
PSF = np.array([1, 4, 8, 16, 19, 15, 10, 7, 1])
PSF = PSF / np.sum(PSF)
A = scipyLinalg.convolution_matrix(PSF, N, mode='same')

delta = 1e-1
m, md = generateMeasurements(A, f, delta)

plt.figure(3)
plt.plot(x, f, 'k', linewidth=linewidth)
plt.plot(x, md, 'r', linewidth=linewidth)
plt.title('Noisy data')
plt.legend(['original signal', 'convolved signal + noise'])
plt.show()

# -------------------------------------------
# Tikhonov regularization L=I, f0=0
# -------------------------------------------
L = np.eye(N)
f0 = np.zeros(N)
alphaMin = -10  # 1e-10
alphaMax = 0  # 1e0
nValues = 20
flagPlotResiduals = True
figureNbr = 4

bestAlpha, residuals = calcMorozov(A, md, L, f0, alphaMin, alphaMax, nValues, delta, flagPlotResiduals, figureNbr)
f_alpha = solveLinearSysTikhonov(A, md, L, f0, bestAlpha)
plotSolution(figureNbr + 1, x, f, md, f_alpha, f'Solution L=I, f0=0, alpha={bestAlpha:.1e}')

# -------------------------------------------
# Tikhonov regularization L= First Derivative, f0=0
# df/dx ~= (f(x)-f(x-h))/h, where h=1 in this example
# -------------------------------------------
L = generateDiffOperator(N)  # the matrix representing the derivative
f0 = np.zeros(N)

figureNbr = 6
bestAlpha, residuals = calcMorozov(A, md, L, f0, alphaMin, alphaMax, nValues, delta, flagPlotResiduals, figureNbr)
f_alpha = solveLinearSysTikhonov(A, md, L, f0, bestAlpha)
plotSolution(figureNbr + 1, x, f, md, f_alpha, f'Solution L=Diff, f0=0, alpha={bestAlpha:.1e}')

# -------------------------------------------
# Tikhonov regularization: filtering solution L=Gaussian high pass filter, f0=0
# -------------------------------------------
stdFilter = 15
impulseResp = np.exp(-np.arange(0, N) ** 2 / stdFilter ** 2)
impulseResp = impulseResp / np.sum(impulseResp)

L_lowpass = toeplitz(impulseResp)
L = np.eye(N) - L_lowpass

f0 = np.zeros(N)

figureNbr = 8
bestAlpha, residuals = calcMorozov(A, md, L, f0, alphaMin, alphaMax, nValues, delta, flagPlotResiduals, figureNbr)
f_alpha = solveLinearSysTikhonov(A, md, L, f0, bestAlpha)
plotSolution(figureNbr + 1, x, f, md, f_alpha, f'Solution L=Gauss HP, f0=0, alpha={bestAlpha:.1e}')

# -------------------------------------------
# Tikhonov regularization L=weight factor, f0=averages
# -------------------------------------------
f0 = np.zeros(N)
n0 = round(N / 15)
n1 = 2 * n0
n2 = 3 * n0
n3 = 4 * n0
n4 = 8 * n0
f0[n1:n2] = 1  # with high confidence
f0[n3:n4] = 1.5  # we know the value 'in average' in the interval

L = np.zeros((N, N))
L[n1:n2, n1:n2] = 10 * np.eye(n2 - n1)  # with high confidence
L[n3:n4, n3:n4] = 1 * np.eye(n4 - n3)  # with lower confidence

figureNbr = 10
bestAlpha, residuals = calcMorozov(A, md, L, f0, alphaMin, alphaMax, nValues, delta, flagPlotResiduals, figureNbr)
f_alpha = solveLinearSysTikhonov(A, md, L, f0, bestAlpha)
plotSolution(figureNbr + 1, x, f, md, f_alpha, f'Solution L=Diagonal Weight, f0=averages, alpha={bestAlpha:.1e}')

# -------------------------------------------
# Tikhonov: add another term to the regularization M=I, g0=0
# -------------------------------------------
M = np.eye(N)
g0 = np.zeros(N)
figureNbr = 12

alpha_val = 1e-2
beta = 1e-2
f_alpha = solveLinearSysTikhonov2(A, md, L, f0, alpha_val, M, g0, beta)
plotSolution(figureNbr + 1, x, f, md, f_alpha, f'Solution L=Diagonal Weight + M=I, f0=averages + g0=0, alpha={alpha_val:.1e}')

# -------------------------------------------
# Tikhonov regularization L=Boundaries f0=constant
# -------------------------------------------
# add offset to the end of the original signal
f[n4:] += 1

# recompute noisy measurements
delta = 1e-1
m, md = generateMeasurements(A, f, delta)

L_diag = np.zeros(N)
wlength = 50
L_diag[:wlength] = 0.5 * (np.cos(np.pi * np.arange(1, wlength + 1) / wlength) + 1)
L_diag[N - wlength:] = L_diag[wlength - 1::-1]  # copy (mirrored) to the end of the diagonal
L = np.diag(L_diag)

plt.figure(14)
plt.plot(np.diag(L), 'k', linewidth=2)
f0 = np.zeros(N)
f0[N - wlength:N] = 1
plt.plot(f0, 'b', linewidth=2)
plt.legend(['diag(L)', 'f0'])
plt.show()

figureNbr = 15
bestAlpha, residuals = calcMorozov(A, md, L, f0, alphaMin, alphaMax, nValues, delta, flagPlotResiduals, figureNbr)
f_alpha = solveLinearSysTikhonov(A, md, L, f0, bestAlpha)
plotSolution(figureNbr + 1, x, f, md, f_alpha, f'Solution L=Borders, f0=constants, alpha={bestAlpha:.1e}')

# -------------------------------------------
# Tikhonov regularization L= piecewise smooth, f0=0
# df/dx ~= (f(x_(i+1))-f(x_i))/h, where h=1 in this example
# -------------------------------------------

LA = generateDiffOperator(n1)
LB = generateDiffOperator(n2 - n1)
LC = generateDiffOperator(n3 - n2)
LD = generateDiffOperator(n4 - n3)
LE = generateDiffOperator(N - n4)

L = scipyLinalg.block_diag(LA, LB, LC, LD, LE)

f0 = np.zeros(N)

figureNbr = 17
bestAlpha, residuals = calcMorozov(A, md, L, f0, alphaMin, alphaMax, nValues, delta, flagPlotResiduals, figureNbr)
f_alpha = solveLinearSysTikhonov(A, md, L, f0, bestAlpha)
plotSolution(figureNbr + 1, x, f, md, f_alpha, f'Solution L=piecewise smooth, f0=0, alpha={bestAlpha:.1e}')
