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
    """
    Load the data of a simulation
    :param simname: name of the simulation
    :return: snapshots, params, snapshots_t, e, m
    """
    simdir = "./simulations/{0}/".format(simname)

    params_file = open(simdir + "params.json", "r")
    params = json.load(params_file)

    snapshots = np.load(simdir + "snapshots.npy")

    results = np.load(simdir + "results.npy")

    e = results[:, 0]
    m = results[:, 1:4]
    # Build steps axis
    snapshots_t = np.load(simdir + "snapshots_t.npy")

    return snapshots, params, snapshots_t, e, m


# Function for computing quantities
def compute_em(snapshots, params):
    """
    Compute energy and magnetization
    """

    J = params["J"]
    h = params["h"]
    T = params["T"]
    snapshots_number = snapshots.shape[0]

    energy = np.zeros(shape=snapshots_number)
    magnetization = np.zeros(shape=(snapshots_number, 3))

    for i in np.arange(0, snapshots_number):
        sys = HeisenbergSystem(snapshots[i, :, :, :, :], J=J, h=h, temperature=T)
        energy[i] = sys.energy
        magnetization[i, :] = sys.magnetization

    return energy, magnetization


def compute_statistics(e, m):
    e_mean = np.mean(e)
    e_rmsd = np.sqrt(np.var(e))
    m_mean = np.mean(m, axis=0)
    m_rmsd = np.sqrt(np.var(m, axis=0))

    stats = dict(e_mean=e_mean, e_rmsd=e_rmsd, m_mean=m_mean, m_rmsd=m_rmsd)
    return stats


# Functions for plotting
def plot_energy(snapshots_t, energy):
    """"
    Plot energy
    """
    fig = plt.figure()
    ax = fig.gca()
    line1, = ax.plot(snapshots_t, energy)
    plt.xlabel("step")
    plt.ylabel("Energy")
    plt.title("Total energy")
    return fig


def plot_magnetization(snapshots_t, magnetization):
    """"
    Plot energy
    """
    fig = plt.figure()
    ax = fig.gca()
    line1, = ax.plot(snapshots_t, magnetization[:, 0], label='Mx')
    line2, = ax.plot(snapshots_t, magnetization[:, 1], label='My')
    line3, = ax.plot(snapshots_t, magnetization[:, 2], label='Mz')
    ax.legend()
    plt.xlabel("step")
    plt.title("Magnetization")
    return fig


def plot_abs_magnetization(snapshots_t, magnetization):
    """"
    Plot energy
    """
    fig = plt.figure()
    ax = fig.gca()
    line1, = ax.plot(snapshots_t, np.abs(magnetization[:, 0]), label='|Mx|')
    line2, = ax.plot(snapshots_t, np.abs(magnetization[:, 1]), label='|My|')
    line3, = ax.plot(snapshots_t, np.abs(magnetization[:, 2]), label='|Mz|')
    ax.legend()
    plt.xlabel("step")
    plt.title("Absolute magnetization")
    return fig

def plot_state(snapshots, n=-1):
    """
    Plot system state
    """
    nx = snapshots.shape[1]
    ny = snapshots.shape[2]
    nz = snapshots.shape[3]

    x, y, z = np.meshgrid(np.arange(0, nx),
                          np.arange(0, ny),
                          np.arange(0, nz))

    state = snapshots[n, :, :, :, :]

    u = np.zeros(shape=(nx, ny, nz))
    v = np.zeros(shape=(nx, ny, nz))
    w = np.zeros(shape=(nx, ny, nz))

    for i, j, k in np.ndindex(nx, ny, nz):
        u[i, j, k], v[i, j, k], w[i, j, k] = sph2xyz(state[i, j, k, 0], state[i, j, k, 1])

    fig = plt.figure()
    ax: Axes3D = fig.gca(projection='3d')
    ax.quiver(x, y, z, u, v, w, pivot='middle')
    plt.show()
