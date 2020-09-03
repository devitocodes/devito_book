import numpy as np
import matplotlib.pyplot as plt

from devito import *

class advection(object):

    def __init__(self, v, L, dt, C, T=None, boundary_conditions=None, time_order=1, space_order=1):
        Nt = int(round(T/float(dt)))-1 if T else None
        dx = v*dt/C
        Nx = int(round(L/dx))+1
        grid = Grid(shape=(Nx, ), extent=(L, ))
        u = TimeFunction(name='u', grid=grid, time_order=time_order, space_order=space_order)
        pde = u.dt + v*u.dx
        stencil = Eq(u.forward, solve(pde, u.forward))

        bc = boundary_conditions if boundary_conditions else []

        self.L = L
        self.Nx = Nx
        self.dt = dt
        self.Nt = Nt
        self.grid = grid
        self.u = u
        self.stencil = stencil
        self.bc = bc

    def initialize(self, case='gaussian', sigma=1.0):
        """Special function for the FECS case."""
        x = np.linspace(0, self.L, self.Nx)
        if case == 'gaussian':
            self.u.data[0] = np.exp(-0.5*((x-self.L/10)/sigma)**2)
        elif case == 'cosinehat':
            self.u.data[0] = np.cos(np.pi*5/self.L*(x - self.L/10)) if x < self.L/5 else 0

    def solve(self, start=0, stop=None, dt=None):
        time_M = stop if stop else self.Nt
        dt = dt if dt else self.dt
        Operator([self.stencil]+self.bc)(time_m=start, time_M=time_M, dt=dt)

v = 1.0
L = 1.0
dt = 0.001
C = 1
T = 0.1

U0 = 0

sigma = 0.02

model = advection(v, L, dt, C, T, boundary_conditions=bcs)
model.initialize(sigma=0.02)

model.solve(stop=0)

from IPython import embed; embed()
