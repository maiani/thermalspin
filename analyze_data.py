#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show simple data of the simulation
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from heisenberg_system import HeisenbergSystem
from math_utils import sph2xyz


def load_data(simname):
    simdir = "./simulations/{0}/".format(simname)

    params_file = open(simdir + "params.json", "r")
    params = json.load(params_file)
    delta_snapshots = params["delta_snp"]

    snapshots = np.load(simdir + "snapshots.npy")
    snapshots_number = snapshots.shape[0]

    # Build steps axis
    snapshots_t = np.load(simdir + "snapshots_t.npy")

    return snapshots, params, snapshots_t


def compute_EM(snapshots, params, snapshots_t):
    """
    Compute energy and magnetization
    """

    J = params["J"]
    h = params["h"]
    T = params["T"]
    snapshots_number = snapshots.shape[0]

    E = np.zeros(shape=(snapshots_number))
    M = np.zeros(shape=(snapshots_number, 3))

    for i in np.arange(0, snapshots_number):
        sys = HeisenbergSystem(snapshots[i, :, :, :, :], J=J, h=h, T=T)
        E[i] = sys.energy
        M[i, :] = sys.magnetization

    return E, M


def plot_energy(snapshots_t, E):
    """"
    Plot energy
    """
    plt.figure()
    plt.plot(snapshots_t, E)
    plt.title("Energy")


def plot_magnetization(snapshots_t, M):
    """"
    Plot energy
    """
    fig = plt.figure()
    ax = fig.gca()
    line1, = ax.plot(snapshots_t, M[:, 0], label='Mx')
    line2, = ax.plot(snapshots_t, M[:, 1], label='My')
    line3, = ax.plot(snapshots_t, M[:, 2], label='Mz')
    ax.legend()
    plt.legend()


def plot_state(snapshots, n=-1):
    """
    Plot system state
    """
    Nx = snapshots.shape[1]
    Ny = snapshots.shape[2]
    Nz = snapshots.shape[3]

    x, y, z = np.meshgrid(np.arange(0, Nx),
                          np.arange(0, Ny),
                          np.arange(0, Nz))

    state = snapshots[n, :, :, :, :]

    u = np.zeros(shape=(Nx, Ny, Nz))
    v = np.zeros(shape=(Nx, Ny, Nz))
    w = np.zeros(shape=(Nx, Ny, Nz))

    for i, j, k in np.ndindex(Nx, Ny, Nz):
        u[i, j, k], v[i, j, k], w[i, j, k] = sph2xyz(state[i, j, k, 0], state[i, j, k, 1])

    fig = plt.figure()
    ax: Axes3D = fig.gca(projection='3d')
    ax.quiver(x, y, z, u, v, w, pivot='middle')
    plt.show()
