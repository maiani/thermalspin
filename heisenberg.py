#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Heisnberg model Monte Carlo simulator
"""

import numpy as np

from utils import sph_dot


class HeisenbergSystem:

    def __init__(self, Nx, Ny, Nz, J=-1):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.J = J

        self.S = np.zeros(shape=(Nx, Ny, Nz, 2))

        for i, j, k in np.ndindex(Nx - 1, Ny - 1, Nz - 1):
            self.S[Nx, Ny, Nz, :] = np.array([0, 0])

    def H(self):
        H = 0

        dotp_counter = 0
        for i, j, k in np.ndindex(self.Nx, self.Ny, self.Nz):
            ii = (i + 1) % self.Nx
            dotp_counter += sph_dot(self.S[i, j, k, 0], self.S[ii, j, k, 0], self.S[i, j, k, 1] - self.S[ii, j, k, 1])
            ii = (i - 1) % self.Nx
            dotp_counter += sph_dot(self.S[i, j, k, 0], self.S[ii, j, k, 0], self.S[i, j, k, 1] - self.S[ii, j, k, 1])

            jj = (j + 1) % self.Ny
            dotp_counter += sph_dot(self.S[i, j, k, 0], self.S[i, jj, k, 0], self.S[i, j, k, 1] - self.S[i, jj, k, 1])
            jj = (j - 1) % self.Ny
            dotp_counter += sph_dot(self.S[i, j, k, 0], self.S[i, jj, k, 0], self.S[i, j, k, 1] - self.S[i, jj, k, 1])

            if self.Nz > 1:
                kk = (k + 1) % self.Nz
                dotp_counter += sph_dot(self.S[i, j, k, 0], self.S[i, j, kk, 0],
                                        self.S[i, j, k, 1] - self.S[i, j, kk, 1])
                kk = (k - 1) % self.Nz
                dotp_counter += sph_dot(self.S[i, j, k, 0], self.S[i, j, kk, 0],
                                        self.S[i, j, k, 1] - self.S[ii, j, kk, 1])

        H += self.J * dotp_counter
        return H


# Main

s = HeisenbergSystem(10, 10, 1)
print("H = ", s.H())
