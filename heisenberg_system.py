#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Heisenberg model Monte Carlo simulator
"""
import numpy as np
from numba import jit

from math_utils import sph_dot, sph2xyz, sph_u_rand


class HeisenbergSystem:
    """
    This class represent a system described by an Heisenberg Hamiltonian, also known as O(3) model
    """

    def __init__(self, state, J, Hz, T):
        self.J = J
        self.Hz = Hz
        self.T = T
        self.beta = 1 / T

        self.state = state
        self.nx = self.state.shape[0]
        self.ny = self.state.shape[1]
        self.nz = self.state.shape[2]
        self.nspin = self.nx * self.ny * self.nz

        # Compute energy and magnetization of the initial state
        self.energy = compute_energy(self.state, self.nx, self.ny, self.nz, J, Hz)
        self.total_magnetization = compute_magnetization(self.state, self.nx, self.ny, self.nz)

    @property
    def magnetization(self):
        """
        The magnetization of the system
        :return: The value of the magnetization
        """
        return self.total_magnetization / self.nspin

    def step(self):
        """
        Evolve the system computing a step of Metropolis-Hastings Monte Carlo.
        It actually calls the non-object oriented function.
        """
        s, e, m = numba_step(self.state, self.nx, self.ny, self.nz, self.J, self.Hz, self.beta, self.energy,
                             self.total_magnetization)
        self.state = s
        self.energy = e
        self.total_magnetization = m


# Compiled functions

@jit(nopython=True, cache=True)
def compute_magnetization(state, nx, ny, nz):
    """
    Compute the total magnetization
    :return: [Mx, My, Mz] vector of mean magnetization
    """

    counter_r = np.zeros(3)

    for i, j, k in np.ndindex(nx, ny, nz):
        r = sph2xyz(state[i, j, k, 0], state[i, j, k, 1])
        counter_r += r

    return counter_r


@jit(nopython=True, cache=True)
def site_energy(i, j, k, state, nx, ny, nz, J, h):
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

    e0 *= - J / 2
    e0 += -h * np.cos(state[i, j, k, 0])

    return e0


@jit(nopython=True, cache=True)
def compute_energy(state, nx, ny, nz, J, h):
    """
    Compute the energy of the system
    :return: The value of the energy
    """

    energy_counter = 0

    for i, j, k in np.ndindex(nx, ny, nz):
        energy_counter += site_energy(i, j, k, state, nx, ny, nz, J, h)

    return np.array(energy_counter)


@jit(nopython=True, cache=True)
def numba_step(state, nx, ny, nz, J, h, beta, energy, total_magnetization):
    """
    Evolve the system computing a step of Metropolis-Hastings Monte Carlo.
    This non OOP function is accelerated trough jit compilation.
    """

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

    e0 *= -J
    e0 += -h * np.cos(state[i, j, k, 0])

    # Generate a new random direction and compute energy due to the spin in the new direction
    r_theta, r_phi = sph_u_rand()

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

    e1 *= -J
    e1 += -h * np.cos(r_theta)

    # Apply Metropolis algorithm
    w = np.exp(beta * (e0 - e1))
    dice = np.random.uniform(0, 1)

    if dice < w:
        energy += (e1 - e0)
        total_magnetization += (sph2xyz(r_theta, r_phi) - sph2xyz(state[i, j, k, 0], state[i, j, k, 1]))
        state[i, j, k, :] = np.array([r_theta, r_phi])

    return state, energy, total_magnetization
