import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.compare import save_diff_image
from scipy import signal as scipySignal
from scipy import linalg as scipyLinalg

showImages=False
saveImages=True
plt.rcParams.update({'font.size': 15})

def generate_ill_conditioned_system(m, n, condition_number):
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

    return A, b, x_true, U, S, V


# build a matrix A with a given condition number
n = 20
condition_number = 1.5e0

A, m, f_true, U, S, V = generate_ill_conditioned_system(n, n, condition_number)
A_pinv = scipyLinalg.pinv(A)
Apinv_norm = np.linalg.norm(A_pinv, 2)

# minimum norm solution
f_plus = A_pinv @ m

# create plots
deltas = np.linspace(0, 5, 100+1)
alphas = np.logspace(2, -2, 32+1)
alphas[-1] = 0.0

# compute the error bound y1=|| f^{+}-f_delta||_2<||A^+||_2 * delta
# this must be a line y1=angCoef1*deltas+linCoef1
angCoef1 = Apinv_norm
linCoef1 = 0
y1 = angCoef1 * deltas + linCoef1

# compute the error bound y2=||f^{+}-f_alpha_delta||_2< ||f^{+}-Ra*m|| + ||Ra||_2 * delta
surface_y2= np.zeros((len(alphas), len(deltas)))
intersectPoints= np.zeros((len(alphas), 2))

for i,alpha in enumerate(alphas):
    print('alpha=%g' % alpha)
    if alpha != 0:
        # Create regularization operator (Tikhonov)
        Ra = scipyLinalg.inv(A.T @ A + alpha * np.eye(n)) @ A.T

        # compute the error bound y2=||f^{+}-f_alpha_delta||_2< ||f^{+}-Ra*m|| + ||Ra||_2 * delta
        # this must be a line y2=angCoef2*deltas+linCoef2
        angCoef2 = np.linalg.norm(Ra, 2)
        linCoef2 = np.linalg.norm(f_plus - Ra @ m, 2)
        surface_y2[i,:] = angCoef2 * deltas + linCoef2

        # intersection point between the two lines
        # y1=angCoef1*deltas+linCoef1
        # y2=angCoef2*deltas+linCoef2
        intersectPoints[i,:] = np.linalg.solve(np.array([[angCoef1, -1], [angCoef2, -1]]), np.array([-linCoef1, -linCoef2]))

    else:
        # if alpha=0, then Ra is the pseudo-inverse and the lines are the same
        linCoef2=0
        intersectPoints[i,:] = np.array([0, 0])
        surface_y2[i,:] = y1

    if i%8==0:
        if showImages:
            fig = plt.figure()
            plt.plot(deltas, y1, '--k', label='A+')
            plt.plot(deltas, surface_y2[i,:], 'b', label='R_\alpha')
            plt.plot(intersectPoints[i,0], intersectPoints[i,1], 'ro', zorder=3)
            plt.plot(deltas[0], linCoef2, 'y*',markersize=12)
            plt.title('$\\alpha$=%0.1e' % alpha)
            plt.xlabel('$\\delta$')
            plt.ylabel('Error bound')
            plt.xlim([deltas[0],deltas[-1]])

            if saveImages:
                plt.savefig('image_alpha_%1.2d.svg' % i, format='svg', bbox_inches='tight', transparent=True)
            else:
                plt.show()
            plt.close()

# 3D plots
y = np.linspace(0, 1, alphas.shape[0])
x = deltas
xG, yG = np.meshgrid(x, y)

ax = plt.figure().add_subplot(projection='3d')
ax.view_init(elev=30, azim=-135, roll=0)
ax.set_yticks([0,0.25,0.5,0.75,1.0], [0,0.1,1,10,100])
ax.set_xlim([deltas[0],deltas[-1]])
ax.set_ylim([0,1])
ax.set_zlim([y1[0],y1[-1]])
ax.set_ylabel('$\\alpha$')
ax.set_xlabel('$\\delta$')
ax.set_zlabel('Error bound')


# add projected 2D plots
for i,alpha in enumerate(alphas):
    if i%8==0:
        ax.plot(x,y[::-1][i]*np.ones_like(y1), y1, '--k', label='A+')
        ax.plot(x,y[::-1][i]*np.ones_like(y1), surface_y2[i,:], 'b', label='R_\alpha')
        ax.plot(intersectPoints[i,0],y[::-1][i], intersectPoints[i,1], 'ro', zorder=4)
        ax.plot(x[0],y[::-1][i], surface_y2[i,0], 'y*',markersize=12)

        if saveImages:
            plt.savefig('image3D_alpha_%1.2d.svg' % i, format='svg', bbox_inches='tight', transparent=True)


# plot y2
ax.plot_surface(xG, yG, surface_y2[::-1,:], cmap='hsv',rstride=1, cstride=1, alpha=0.5, zorder=2) #surface only
ax.plot_surface(xG, yG, surface_y2[::-1,:], edgecolor='royalblue', lw=0.5, rstride=8, cstride=50, alpha=0, zorder=3) #edges only

plt.savefig('image3D_y2.svg', format='svg', bbox_inches='tight', transparent=True)

#plot y1
surface_y1 = np.matlib.repmat(y1, len(alphas), 1)
ax.plot_surface(xG, yG, surface_y1, color='gray', alpha=0.4,zorder=1)

#plot intersections
ax.plot(intersectPoints[:,0],y[::-1], intersectPoints[:,1], 'r', lw=1.5, zorder=4)

plt.savefig('image3D_y1.svg', format='svg', bbox_inches='tight', transparent=True)

#plot pcolor
ax = plt.figure().add_subplot()
ax.pcolormesh(xG, yG, surface_y2[::-1,:], cmap='hsv',shading='gouraud', vmin=surface_y2.min(), vmax=surface_y2.max())
ax.plot(intersectPoints[:,0],y[::-1], 'r', lw=1.5, zorder=4)
ax.set_xlabel('$\\delta$')
ax.set_ylabel('$\\alpha$')
ax.set_yticks([0,0.25,0.5,0.75,1.0], [0,0.1,1,10,100])

plt.savefig('image_pcolor.png', format='png', bbox_inches='tight', transparent=True)

plt.show()
