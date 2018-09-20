#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show simple data of the simulation
"""

import json

import matplotlib.pyplot as plt
import numpy as np

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
    t = np.arange(0, snapshots_number) * delta_snapshots

    return snapshots, params, t


def compute_EM(snapshots, params, t):
    # Compute energy and magnetization

    J = params["J"]
    h = params["h"]
    T = params["T"]
    snapshots_number = snapshots.shape[0]

    E = np.zeros(shape=(snapshots_number))
    M = np.zeros(shape=(snapshots_number, 3))

    for i in np.arange(0, snapshots_number):
        sys = HeisenbergSystem(snapshots[i, :, :, :, :], J=J, h=h, T=T)
        E[i] = sys.H
        M[i, :] = sys.M

    return E, M


def plot_EM(t, E, M):
    # Plot energy
    plt.figure()
    plt.plot(t, E)
    plt.title("Energy")

    # Plot magnetization
    plt.figure()
    plt.plot(t, M[:, 0])
    plt.title("Mx")

    plt.figure()
    plt.plot(t, M[:, 1])
    plt.title("My")

    plt.figure()
    plt.plot(t, M[:, 2])
    plt.title("Mz")


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

    S = snapshots[n, :, :, :, :]

    u = np.zeros(shape=(Nx, Ny, Nz))
    v = np.zeros(shape=(Nx, Ny, Nz))
    w = np.zeros(shape=(Nx, Ny, Nz))

    for i, j, k in np.ndindex(Nx, Ny, Nz):
        u[i, j, k], v[i, j, k], w[i, j, k] = sph2xyz(S[i, j, k, 0], S[i, j, k, 1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(x, y, z, u, v, w, pivot='middle')
    ax.set_zlim(-2, 2)
    plt.show()


if __name__ == "__main__":
    snapshots, params, t = load_data("sim_0")
    E, M = compute_EM(snapshots, params, t)
    plot_EM(t, E, M)
    plot_state(snapshots)
