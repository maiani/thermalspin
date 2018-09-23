#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Heisenberg model Monte Carlo simulator
"""
import numpy as np
from numba import jit

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
        self.nspin = self.nx * self.ny * self.nz
        self.energy = self.compute_energy()
        self.total_magnetization = self.compute_magnetization()

    @property
    def magnetization(self):
        return self.total_magnetization / self.nspin

    def compute_energy(self):
        """
        Compute the energy of the system
        :return: The value of the energy
        """
        energy_counter = 0

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
                                        self.state[i, j, k, 1] - self.state[i, j, kk, 1])

            ext_field_counter += np.cos(self.state[i, j, k, 0])

        energy_counter += -self.J * dotp_counter
        energy_counter += -self.h * ext_field_counter

        return np.array(energy_counter)

    def compute_magnetization(self):
        """
        Compute the mean magnetization
        :return: [Mx, My, Mz] vector of mean magnetization
        """
        counter_r = np.zeros(3)

        for i, j, k in np.ndindex(self.nx, self.ny, self.nz):
            r = sph2xyz(self.state[i, j, k, 0], self.state[i, j, k, 1])
            counter_r += r

        return counter_r

    def step(self):
        """
        Evolve the system computing a step of Metropolis-Hastings Monte Carlo
        """
        s, e, m = numba_step(self.state, self.nx, self.ny, self.nz, self.J, self.h, self.beta, self.energy,
                             self.total_magnetization)
        self.state = s
        self.energy = e
        self.total_magnetization = m


@jit(nopython=True)
def numba_step(state, nx, ny, nz, J, h, beta, energy, total_magnetization):
    # Select a random spin in the system
    i = np.random.randint(0, nx)
    j = np.random.randint(0, ny)
    k = np.random.randint(0, nz)

    # Compute the energy due to that spin
    e0 = 0
    ii = (i + 1) % nx
    e0 += sph_dot(state[i, j, k, 0], state[ii, j, k, 0],
                  state[i, j, k, 1] - state[ii, j, k, 1])
    ii = (i - 1) % nx
    e0 += sph_dot(state[i, j, k, 0], state[ii, j, k, 0],
                  state[i, j, k, 1] - state[ii, j, k, 1])
    jj = (j + 1) % ny
    e0 += sph_dot(state[i, j, k, 0], state[i, jj, k, 0],
                  state[i, j, k, 1] - state[i, jj, k, 1])
    jj = (j - 1) % ny
    e0 += sph_dot(state[i, j, k, 0], state[i, jj, k, 0],
                  state[i, j, k, 1] - state[i, jj, k, 1])

    if nz > 1:
        kk = (k + 1) % nz
        e0 += sph_dot(state[i, j, k, 0], state[i, j, kk, 0],
                      state[i, j, k, 1] - state[i, j, kk, 1])
        kk = (k - 1) % nz
        e0 += sph_dot(state[i, j, k, 0], state[i, j, kk, 0],
                      state[i, j, k, 1] - state[i, j, kk, 1])

    e0 *= -2 * J
    e0 += -h * np.cos(state[i, j, k, 0])

    # Generate a new random direction and compute energy due to the spin in the new direction
    r_theta, r_phi = rand_sph()

    e1 = 0
    ii = (i + 1) % nx
    e1 += sph_dot(r_theta, state[ii, j, k, 0],
                  r_phi - state[ii, j, k, 1])
    ii = (i - 1) % nx
    e1 += sph_dot(r_theta, state[ii, j, k, 0],
                  r_phi - state[ii, j, k, 1])
    jj = (j + 1) % ny
    e1 += sph_dot(r_theta, state[i, jj, k, 0],
                  r_phi - state[i, jj, k, 1])
    jj = (j - 1) % ny
    e1 += sph_dot(r_theta, state[i, jj, k, 0],
                  r_phi - state[i, jj, k, 1])

    if nz > 1:
        kk = (k + 1) % nz
        e1 += sph_dot(r_theta, state[i, j, kk, 0],
                      r_phi - state[i, j, kk, 1])
        kk = (k - 1) % nz
        e1 += sph_dot(r_theta, state[i, j, kk, 0],
                      r_phi - state[i, j, kk, 1])

    e1 *= -2 * J
    e1 += -h * np.cos(r_theta)

    # Apply Metropolis algorithm
    w = np.exp(beta * (e0 - e1))
    dice = np.random.uniform(0, 1)

    if dice < w:
        energy += (e1 - e0)
        total_magnetization += (sph2xyz(r_theta, r_phi) - sph2xyz(state[i, j, k, 0], state[i, j, k, 1]))
        state[i, j, k, :] = np.array([r_theta, r_phi])

    return state, energy, total_magnetization
