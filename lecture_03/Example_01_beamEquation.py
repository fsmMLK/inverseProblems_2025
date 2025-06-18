import sympy
import numpy as np
from sympy.matrices import Matrix, eye, zeros, ones, diag

s,t = sympy.symbols('s,t')

# ===================================
# legendre polynomials of degrees 0 to 3 for w(t) in the interval [0,1]
# ===================================
maxOrder=4
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

# ===================================
# greens functions
# ===================================
# solution here: https://trace.tennessee.edu/cgi/viewcontent.cgi?article=2803&context=utk_gradthes
# example 3.3
GL=(s**2)/2*(-s/3+t)  #s<t
GR=(t**2)/2*(-t/3+s)  #s>t

# ===================================
# load
# ===================================
g=1;
EI=1;

# ===================================
# assemble the system
# ===================================
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

# build the right-hand side
b=sympy.matrices.zeros(1,dimBase)

#integrations over s
for i in range(dimBase):
    b[i] = sympy.integrate(psiBase[i]*g, (s, 0, 1))/EI

if True:
    #numpy solution
    Anp = np.array(A.tolist()).astype(np.float64)
    bnp = np.array(b.tolist()).astype(np.float64)
    xnp = np.linalg.solve((Anp), bnp.T).flatten()
    print(np.linalg.cond(Anp))

    #regularized solution
    Lamb=1e-0
    M=Anp.T@Anp+Lamb*np.eye(Anp.shape[1])
    N=Anp.T@bnp.T
    xnpReg= np.linalg.lstsq(M, N, rcond=None)[0].flatten()

    x=xnpReg
else:
    #symbolic solution
    x=sympy.linsolve([A,b])
    x=list(list(x)[0])  # extract the solution from the linsolve object
    x=[ elem.evalf() for elem in x]

# ===================================
#build solution from the
# ===================================
w=0
for i in range(dimBase):
    w+= x[i]*psiBase[i]

w=w.evalf()
w= sympy.simplify(w)

coefs= np.array(sympy.Poly(w, s).coeffs())
#coefs/=coefs[0]
print('Coefficient : ', coefs)
print('Coefficient  normalized: ', coefs/coefs[0])
print('fim!')