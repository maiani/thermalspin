#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Heisenberg model Monte Carlo simulator
"""

import numpy as np

from math_utils import sph_dot, sph2xyz, rand_sph


class HeisenbergSystem:

    def __init__(self, state, J, h, T):
        self.J = J
        self.h = h
        self.T = T
        self.beta = 1 / T

        self.state = state
        self.nx = self.state.shape[0]
        self.ny = self.state.shape[1]
        self.nz = self.state.shape[2]

    @property
    def energy(self):
        """
        Compute the energy of the system
        :return: The value of the energy
        """
        H = 0

        dotp_counter = 0
        ext_field_counter = 0

        for i, j, k in np.ndindex(self.nx, self.ny, self.nz):
            ii = (i + 1) % self.nx
            dotp_counter += sph_dot(self.state[i, j, k, 0], self.state[ii, j, k, 0],
                                    self.state[i, j, k, 1] - self.state[ii, j, k, 1])
            ii = (i - 1) % self.nx
            dotp_counter += sph_dot(self.state[i, j, k, 0], self.state[ii, j, k, 0],
                                    self.state[i, j, k, 1] - self.state[ii, j, k, 1])

            jj = (j + 1) % self.ny
            dotp_counter += sph_dot(self.state[i, j, k, 0], self.state[i, jj, k, 0],
                                    self.state[i, j, k, 1] - self.state[i, jj, k, 1])
            jj = (j - 1) % self.ny
            dotp_counter += sph_dot(self.state[i, j, k, 0], self.state[i, jj, k, 0],
                                    self.state[i, j, k, 1] - self.state[i, jj, k, 1])

            if self.nz > 1:
                kk = (k + 1) % self.nz
                dotp_counter += sph_dot(self.state[i, j, k, 0], self.state[i, j, kk, 0],
                                        self.state[i, j, k, 1] - self.state[i, j, kk, 1])
                kk = (k - 1) % self.nz
                dotp_counter += sph_dot(self.state[i, j, k, 0], self.state[i, j, kk, 0],
                                        self.state[i, j, k, 1] - self.state[ii, j, kk, 1])
            ext_field_counter += np.cos(self.state[i, j, k, 0])

        H += -self.J * dotp_counter
        H += -self.h * ext_field_counter

        return H

    @property
    def magnetization(self):
        """
        Compute the mean magnetization
        :return: [Mx, My, Mz] vector of mean magnetization
        """
        counter_x = 0
        counter_y = 0
        counter_z = 0

        for i, j, k in np.ndindex(self.nx, self.ny, self.nz):
            x, y, z = sph2xyz(self.state[i, j, k, 0], self.state[i, j, k, 1])
            counter_x += x
            counter_y += y
            counter_z += z

        return np.array([counter_x, counter_y, counter_z]) / self.nx / self.ny / self.nz

    def step(self):
        """
        Evolve the system computing a step of Metropolis-Hastings Monte Carlo
        """

        i = np.random.randint(0, self.nx)
        j = np.random.randint(0, self.ny)
        k = np.random.randint(0, self.nz)

        r_theta, r_phi = rand_sph()

        E0 = 0
        ii = (i + 1) % self.nx
        E0 += sph_dot(self.state[i, j, k, 0], self.state[ii, j, k, 0],
                      self.state[i, j, k, 1] - self.state[ii, j, k, 1])
        ii = (i - 1) % self.nx
        E0 += sph_dot(self.state[i, j, k, 0], self.state[ii, j, k, 0],
                      self.state[i, j, k, 1] - self.state[ii, j, k, 1])
        jj = (j + 1) % self.ny
        E0 += sph_dot(self.state[i, j, k, 0], self.state[i, jj, k, 0],
                      self.state[i, j, k, 1] - self.state[i, jj, k, 1])
        jj = (j - 1) % self.ny
        E0 += sph_dot(self.state[i, j, k, 0], self.state[i, jj, k, 0],
                      self.state[i, j, k, 1] - self.state[i, jj, k, 1])

        if self.nz > 1:
            kk = (k + 1) % self.nz
            E0 += sph_dot(self.state[i, j, k, 0], self.state[i, j, kk, 0],
                          self.state[i, j, k, 1] - self.state[i, j, kk, 1])
            kk = (k - 1) % self.nz
            E0 += sph_dot(self.state[i, j, k, 0], self.state[i, j, kk, 0],
                          self.state[i, j, k, 1] - self.state[ii, j, kk, 1])

        E0 *= -self.J
        E0 += -self.h * np.cos(self.state[i, j, k, 0])

        E1 = 0
        ii = (i + 1) % self.nx
        E1 += sph_dot(r_theta, self.state[ii, j, k, 0],
                      r_phi - self.state[ii, j, k, 1])
        ii = (i - 1) % self.nx
        E1 += sph_dot(r_theta, self.state[ii, j, k, 0],
                      r_phi - self.state[ii, j, k, 1])
        jj = (j + 1) % self.ny
        E1 += sph_dot(r_theta, self.state[i, jj, k, 0],
                      r_phi - self.state[i, jj, k, 1])
        jj = (j - 1) % self.ny
        E1 += sph_dot(self.state[i, j, k, 0], self.state[i, jj, k, 0],
                      r_phi - self.state[i, jj, k, 1])

        if self.nz > 1:
            kk = (k + 1) % self.nz
            E1 += sph_dot(r_theta, self.state[i, j, kk, 0],
                          r_phi - self.state[i, j, kk, 1])
            kk = (k - 1) % self.nz
            E1 += sph_dot(r_theta, self.state[i, j, kk, 0],
                          r_phi - self.state[ii, j, kk, 1])

        E1 *= -self.J
        E1 += -self.h * np.cos(r_theta)

        w = np.exp(self.beta * (E0 - E1))
        dice = np.random.uniform()

        if dice < w:
            self.state[i, j, k, :] = np.array([r_theta, r_phi])
