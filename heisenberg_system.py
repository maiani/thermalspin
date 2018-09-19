#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Heisnberg model Monte Carlo simulator
"""

import copy

import numpy as np

from math_utils import sph_dot, sph2xyz


class HeisenbergSystem:

    def __init__(self, S=np.zeros(shape=(1, 1, 1, 2)), J=1, h=0, T=1e-12):
        self.J = J
        self.h = h
        self.T = T
        self.beta = 1 / T

        self.S = S
        self.Nx = self.S.shape[0]
        self.Ny = self.S.shape[1]
        self.Nz = self.S.shape[2]

    def build_aligned_system(self, Nx, Ny, Nz):
        """
        Generate a lattice of spins aligned upward z
        :param Nx:
        :param Ny:
        :param Nz:
        :return:
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.S = np.zeros(shape=(Nx, Ny, Nz, 2))
        # for i, j, k in np.ndindex(Nx - 1, Ny - 1, Nz - 1):
        #     self.S[i, j, k, :] = np.array([0, 0])

    @property
    def H(self):
        """
        Compute the energy of the system
        :return: The value of the energy
        """
        H = 0

        dotp_counter = 0
        ext_field_counter = 0

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
            ext_field_counter += np.cos(self.S[i, j, k, 0])

        H += -self.J * dotp_counter
        H += -self.h * ext_field_counter

        return H

    @property
    def M(self):
        counter_x = 0
        counter_y = 0
        counter_z = 0

        for i, j, k in np.ndindex(self.Nx, self.Ny, self.Nz):
            x, y, z = sph2xyz(self.S[i, j, k, 0], self.S[i, j, k, 1])
            counter_x += x
            counter_y += y
            counter_z += z

        return np.array([counter_x, counter_y, counter_z]) / self.Nx / self.Ny / self.Nz

    def step(self):
        """
        Evolve the system computing a step of Metropolis-Hastings Monte Carlo
        """
        x = np.random.randint(0, self.Nx)
        y = np.random.randint(0, self.Ny)
        z = np.random.randint(0, self.Nz)

        r_theta = np.random.uniform(0, np.pi)
        r_phi = np.random.uniform(0, 2 * np.pi)

        hs_1 = copy.deepcopy(self)

        hs_1.S[x, y, z, :] = np.array([r_theta, r_phi])

        w = np.exp(self.beta * (self.H - hs_1.H))
        dice = np.random.uniform()

        if dice < w:
            self.S = hs_1.S
