"""
Solving 1D heat equation using Crank-Nicolson method with tridiagonal matrix algorithm.
ref:
    1) https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method
    2) https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    3) https://struggler.tistory.com/54

"""


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os, sys

alpha = 1       # k(heat conductivity)/rho(density) * c (heat capacity)
dt = 0.0001      # time step
dx = 0.01       # length step
length = 1      # length of the rod


time = 1       # future time we want to predict
t = np.arange(0, time, dt)      # points along time axis
m = len(t)


l = np.arange(0, length, dx)    # points along length axis
n = len(l)

T = np.ones(n)*500   # initial condition
bc = [500, 0]        # boundary condition

r = (alpha*dt)/(2*dx**2)


def crank_1d(T,r,bc):
    n = len(T)
    d = np.ones(n)
    d[1] = r*T[2] + (1-2*r)*T[1] + r*T[0] + r*bc[0]
    d[-2] = r*T[-1] + (1-2*r)*T[-2] + r*T[-3] + r*bc[-1]
    for i in range(2,n-2):
        d[i] = r*T[i+1] + (1-2*r)*T[i] + r*T[i-1]


    e = np.ones(n)
    f = np.ones(n)

    e[1] = r/(1+2*r)
    f[1] = d[1]/(1+2*r)

    for i in range(2, n-1):
        temp = 1 + 2*r - r*e[i-1]
        e[i] = r / temp
        f[i] = (r*f[i-1] + d[i]) / temp
    T_next = np.empty(n)
    T_next[0] = bc[0]
    T_next[-1] = bc[-1]
    T_next[-2] = (r*f[-3] + d[-2]) / ( 1 + (2*r) - (r*e[-3]) )
    for i in range(n-3, 0, -1):
        T_next[i] = (e[i]*T[i+1]) + f[i]
    return T_next

nPlots = 100        #number of plots to animate

T_grid = np.empty([nPlots+1, n])      # to store data for animation
T_grid[0] = T
c = 1
for i in range(m):
    T = crank_1d(T, r, bc)
    if i%(m//nPlots) == 0:
        T_grid[c] = T
        c += 1

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,510)
line, = ax.plot(l, T)

def animation_frame(i):
    label = 't = {:.2f}'.format(i*(m//nPlots)*dt)
    line.set_ydata(T_grid[i])
    ax.set_title(label)
    return line, ax

anim = FuncAnimation(fig, animation_frame, repeat=True, frames=len(T_grid), interval=100)

fn = '1d-heat-wquation'
anim.save(fn+'.mp4',writer='ffmpeg',fps=10)

plt.show()


