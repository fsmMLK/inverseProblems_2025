import sympy
import numpy as np
from sympy.matrices import Matrix, eye, zeros, ones, diag
import matplotlib.pyplot as plt

"""We will use the symbolic toolbox to compute the integrals exactly. First we need to declare the symbolic variables s and t"""

s,t = sympy.symbols('s,t')

"""First we declare the basis for $\phi_j(t)$ and $\psi_i(s)$. We will choose a polynomial bases, both 4th order. This choise is based on the fact that the solution to the problem we are trying to solve is a polynomial of the fourth order. With this choise we guarantee that the solution will be (in theory) exact.

In princile we could use the canonical basis $\{1, x, x^2,x^3,x^4\}$ but these are not orthogonal in the interval $[0,1]$. We will use [Legendre polynomials](https://https://en.wikipedia.org/wiki/Legendre_polynomials). These are orthogonal.
"""

# ===================================
# legendre polynomials of degrees 0 to 4 in the interval [0,1]
# ===================================
maxOrder=4 # maximum order of the polynomial basis
dimBase=maxOrder+1
phiBase=[]
psiBase=[]
for i in range(dimBase):

    #solution space
    phi=sympy.legendre(i,t) # Legendre orthogonal polynomial of degree i in [-1,1]
    phi = phi.subs(t, 2*t-1)  # map in [0,1]
    phi= phi/sympy.sqrt(sympy.integrate(phi*phi, (t, 0, 1)))  #normalize
    phi = sympy.simplify(phi)
    phiBase.append(phi)

    #input
    psi=sympy.legendre(i,s) # Legendre orthogonal polynomial of degree i in [-1,1]
    psi = psi.subs(s, 2*s-1)  # map in [0,1]
    psi/=sympy.sqrt(sympy.integrate(psi*psi, (s, 0, 1)))  #normalize
    psi = sympy.simplify(psi)
    psiBase.append(psi)

"""Next we define the Green's function for the fouth order ODE as developed in class. You can read more about it [here](https://trace.tennessee.edu/cgi/viewcontent.cgi?article=2803&context=utk_gradthes)."""

# ===================================
# Green's function for the problem
# ===================================
# solution here: https://trace.tennessee.edu/cgi/viewcontent.cgi?article=2803&context=utk_gradthes
# example 3.3
GL=(s**2)/2*(-s/3+t)  #s<t
GR=(t**2)/2*(-t/3+s)  #s>t

"""We are solving the problem for $q(x)=1$. And assuming flexural properties  $EI=1$"""

# ===================================
# load and elastic properties
# ===================================
q0=2; # constant load
EI=1; # flexural rigidity

"""Now we apply the Galerkin method and create the system $Ax=b$"""

# ===================================
# assemble the system's matrix and right-hand side
# ===================================

# 1- build the system matrix
A=sympy.matrices.zeros(dimBase,dimBase)

#integrations over t
IR=sympy.matrices.zeros(1,dimBase)
IL=sympy.matrices.zeros(1,dimBase)
for j in range(dimBase):
    f= GR*phiBase[j]
    IR[j]=sympy.simplify(sympy.integrate(GR*phiBase[j], (t,0,s)))
    IL[j]=sympy.simplify(sympy.integrate(GL*phiBase[j], (t,s,1)))

#integrations over s
for i in range(dimBase):
    for j in range(dimBase):
        A[i,j] = sympy.integrate(psiBase[i]*(IR[j]+IL[j]), (s,0,1))

# 2- build the right-hand side
b=sympy.matrices.zeros(1,dimBase)

#integrations over s
for i in range(dimBase):
    b[i] = sympy.integrate(psiBase[i]*q0/EI, (s, 0, 1))

"""Now we solve the linear system. Here we can choose symbolic solution or numeric solution. In this example there is no difference in the solutions between them.

We can optionally use regularization (Tikhonov) with the numeric solution.
"""

# ===================================
# Solve the system
# ===================================
numpySolution = True  # set to False for numpy solution
regularize=False # add regularization to the solution. for numpy solution only
alpha_tik=1e-0

if numpySolution:
    #numpy solution
    Anp = np.array(A.tolist()).astype(np.float64)
    bnp = np.array(b.tolist()).astype(np.float64)
    xnp = np.linalg.solve((Anp), bnp.T).flatten()
    print("condition number of A: ", np.linalg.cond(Anp))
    x = xnp
    if regularize:
        #regularized solution
        M=Anp.T@Anp+alpha_tik*np.eye(Anp.shape[1])
        N=Anp.T@bnp.T
        xnpReg= np.linalg.lstsq(M, N, rcond=None)[0].flatten()

        x=xnpReg
else:
    #symbolic solution
    x=sympy.linsolve([A,b])
    x=list(list(x)[0])  # extract the solution from the linsolve object
    x=[ elem.evalf() for elem in x]

"""Now we build the solution $w(t)=\sum \zeta_i \phi(t)$"""

# ===================================
# build solution function w(t)
# ===================================
w=0
for i in range(dimBase):
    w+= x[i]*phiBase[i]

w=w.evalf()
w= sympy.simplify(w)

"""We plot the numeric solution together with the exact solition.
The exact solution is $w(x)=\frac{q_0}{24EI}(x^4-4Lx^3+6L^2x^2)$
"""

coefs= np.array(sympy.Poly(w, t).coeffs())
coefs_exact=q0/(24*EI)*np.array([1, -4, 6,0,0])
print('Exact Solution: (%f) x^4 + (%f) x^3 + (%f)x^2 + (%f)x + (%f)' % (1/24, -4/24, 6/24, 0, 0))
print('Coefficient : ', coefs)
print('Relative error (%) ', np.abs(np.divide(coefs[:3]-coefs_exact[:3],coefs_exact[:3]))*100)
print('fim!')

"""Large errors? Go back to the linear system solution and activate the regularization."""

# ===================================
# plot the solution in the interval [0,1]
# ===================================

x=np.linspace(0,1,100)
solution=np.array([w.subs(t,elem).evalf() for elem in x])
exact_solution=q0/(24*EI)*(x**4-4*x**3+6*x**2)   # *np.array

# plot solution (we plot negative because w>0 points down)
plt.figure()
plt.plot(x,-solution)
plt.plot(x,-exact_solution)
plt.legend(['numerical','exact'])
plt.xlabel('x [m]')
plt.ylabel('w(x) [m]')
plt.grid()
plt.show()
