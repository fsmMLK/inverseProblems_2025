import numpy as np
from numpy import linalg as numpyLinalg
import matplotlib.pyplot as plt

def generate_ill_conditioned_system(m,n, condition_number):
    """
    Generate an ill-conditioned linear system Ax = b
    A is mxn
    x is nx1
    b is mx1
    with a specified condition number.
    """
    # Random orthogonal matrix U
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))

    # Create a mxn diagonal matrix with desired condition number
    # The singular values will be spaced logarithmically between 1 and the condition number
    s = np.logspace(0, -np.log10(condition_number), min(m, n))
    S = np.zeros((m, n))
    # Fill the diagonal of S with singular values
    for i in range(min(m, n)):
        S[i, i] = s[i]

    # Construct the ill-conditioned matrix
    A = U @ S @ V.T
    x_true = np.random.randn(n)
    b = A @ x_true

    return A, b, x_true,U,S,V

def compSolution(x,x_true):
    """
    Compute the relative error between the solution x and the true solution x_true
    """
    return numpyLinalg.norm(x - x_true) / numpyLinalg.norm(x_true)


# Step 1 and 2: ill-conditioned system

# Parameters
n_row = 7 # number of equations
n_col = 5 # number of unknowns
condition_numbers = [1e2, 1e4, 1e6, 1e8, 1e10] # condition numbers to test

#generate an ill-conditioned system with specified condition number
A, m, f_true,U,S,V = generate_ill_conditioned_system(n_row,n_col, condition_numbers[4])
A
#add noise to the right-hand side
delta = 1e-4

eps = np.random.normal(0, delta, size=n_row)
m_delta = m + eps

if n_row==n_col:
    #1 - naive inversion
    x_naive = np.linalg.solve(A, m_delta)
    print("Naive solution relative error: %f" % compSolution(x_naive, f_true))

# Step 3: Standard least squares
# this will return the minimum norm solution if the system is underdetermined
# https://numpy.org/doc/2.1/reference/generated/numpy.linalg.lstsq.html
f_lstsq = numpyLinalg.lstsq(A, m_delta)[0]
print("Least squares solution relative error: %f" % compSolution(f_lstsq, f_true))

# Step 4: Moore-Penrose pseudoinverse (manually built)
alpha= 1e-7 # singula value threshold
S_plus= np.zeros_like(S.T)
# Fill the diagonal of S with singular values
for i in range(min(n_row, n_col)):
    if S[i, i] > alpha:
        S_plus[i, i] = 1 / S[i, i]  # Inverse of singular values above threshold
    else:
        S_plus[i, i] = 0  # Set to zero if below threshold
A_pinv = V @ S_plus @ U.T  # Pseudoinverse of A
f_pinv_manual = A_pinv @ m_delta
print("Moore-Penrose pseudoinverse (manually built) solution relative error: %f" % compSolution(f_pinv_manual, f_true))

# Step 5: Moore-Penrose pseudoinverse (numpy built-in)
f_pinv = numpyLinalg.pinv(A,rcond=alpha) @ m_delta
print("Moore-Penrose pseudoinverse (numpy built-in) solution relative error: %f" % compSolution(f_pinv, f_true))


