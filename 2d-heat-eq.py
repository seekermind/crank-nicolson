import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
# from matplotlib import cm
# from mpl_toolkits import mplot3d
from scipy import sparse
import numpy as np

alpha = 1e-4       # k(heat conductivity)/rho(density) * c (heat capacity)
dt = 100      # time step
xPoints = 10      # length step
yPoints = xPoints
length = 1      # length of the surface
width = 1       # width of the surface
dx = length/xPoints
dy = width/yPoints

X, Y = np.meshgrid(np.linspace(0, length, xPoints), np.linspace(0, width, yPoints))
T = np.full([yPoints, xPoints], 200)    # Temperature mesh grid


"""

initial condition

[[100 100 100 ... 100 100 100]
 [100 200 200 ... 200 200 100]
 [100 200 200 ... 200 200 100]
 ...
 [100 200 200 ... 200 200 100]
 [100 200 200 ... 200 200 100]
 [100 100 100 ... 100 100 100]]


"""
initialTemp = 100
T[:, 0] = np.full(yPoints, initialTemp)
T[:, -1] = np.full(yPoints, initialTemp)
T[0] = np.full(xPoints, initialTemp)
T[-1] = np.full(xPoints, initialTemp)

r = (alpha*dt)/(2*dx**2)


def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc


def crank_2d(T, r):
    r = r/2
    a = (1+2*r)
    b = (1-2*r)
    n = (len(T)-2) *(len(T[0])-2)
    N = len(T[0]) - 2
    Lx = sparse.diags([a, -r, -r], [0, -1, 1], shape=(N, N))
    Ix = sparse.identity(N)
    D = sparse.kron(Ix, Lx)
    mainDiagonal = D.diagonal()
    lowerDiagonal = D.diagonal(-1)
    upperDiagonal = D.diagonal(1)
    

    B = sparse.diags([b, r, r], [0, -N, N], shape=(n, n)).toarray()

    # plt.spy(B)
    # plt.show()
    bc = np.zeros(n)
    
    T_result = np.copy(T)

    for iteration in range(2):
        u = T_result[1:-1, 1:-1].flatten()
        bc[0] = T_result[1][0] + T_result[0][1]
        bc[1:N-1] = T_result[0,2:-2]
        bc[N-1] = T_result[0][-2] + T_result[1][-1]

        for j in range(2, N):
            bc[(j-1)*N] = T_result[j,0]
            bc[(j*N)-1] = T_result[j,-1]

        bc[-N] = T_result[-2][0] + T_result[-1][1]
        bc[-N+1:-1] = T_result[-1, 2:-2]
        bc[-1] = T_result[-1][-2]+T_result[-2][-1]

        d = B.dot(u)+r*bc
        T_result[1:-1, 1:-1] = TDMAsolver(lowerDiagonal, mainDiagonal, upperDiagonal, d).reshape(-1, N)
        T_result = T_result.transpose()
    return T_result

frn = 30
T_frames = np.array([T]*frn)
c = 0

timesteps = 25
for i in range(timesteps):
    T = crank_2d(T, r)
    if i%(timesteps//frn + 1) == 0:
        T_frames[c] = T 
        c += 1

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, zarray[frame_number,:,:], cmap="magma")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot = [ax.plot_surface(X, Y, T_frames[0, :,:], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(99,201)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(T_frames, plot), interval=300)

fn = 'plot_surface_animation_funcanimation'
ani.save(fn+'.mp4',writer='ffmpeg',fps=10)

plt.show()
