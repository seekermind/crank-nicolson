import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# input parameter
den = 8880.0
cp = 386.0
cond = 398.0
temp_bc = 100.0
temp_init = 0.0
lx = 1.0
nx = 101
tend = 20000.0
dt = 1.0
tout = 100.0
eps = 1.0e-2
itermax = 1000

alpha = cond / (den * cp)
dx = lx / (nx - 1)
nt = int(tend / dt)
nout = int(tout / dt)

ck = alpha * dt / (dx * dx)

#initial condition
temp = np.full(nx, temp_init) 
time = 0.0

temp_old = np.zeros(nx)

# Boundary condition
temp[0] = temp_bc # Dirichlet @ x=0
temp[nx-1] = temp[nx-2] # Neumann @ x=Lx

# graph data array
ims = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

gx = np.zeros(nx)
for i in range(nx):
    gx[i] = i * dx

# time loop
for n in range(1, nt+1):
    # Crank-Nicolson with Gauss-Seidel
    for i in range(nx):
        temp_old[i] = temp[i]

    for iter in range(itermax):
        resd = 0.0
        for i in range(1, nx-1):
            tp = temp[i]
            temp[i] = 1.0 / (1.0 + ck) * ((1.0 - ck) * temp_old[i] \
                + 0.5 * ck * temp_old[i+1] + 0.5 * ck * temp_old[i-1] \
                + 0.5 * ck * temp[i+1] + 0.5 * ck * temp[i-1])
            resd += abs(temp[i] - tp)

        if resd <= eps:
            break

    # Boundary condition
    temp[0] = temp_bc # Dirichlet @ x=0
    temp[nx-1] = temp[nx-2] # Neumann @ x=Lx

    time += dt

    if n % nout == 0:   
        print('n: {0:7d}, time: {1:8.1f}, temp: {2:10.6f}, iter: {3:4d}'.format(n, time, temp[nx-1], iter))
        im_line = ax.plot(gx, temp, 'b')
        im_time = ax.text(0, 110, 'Time = {0:8.1f} [s]'.format(time))
        ims.append(im_line + [im_time])

# graph plot
ax.set_xlabel('x [m]')
ax.set_ylabel('Temperature [C]')
ax.grid()
anm = animation.ArtistAnimation(fig, ims, interval=50)
anm.save('animation.gif', writer='pillow')
plt.show()
