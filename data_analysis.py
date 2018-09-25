#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show simple data of the simulation
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from heisenberg_system import HeisenbergSystem
from math_utils import sph2xyz

WORKING_DIRECTORY = "./simulations/"


def load_results(simname):
    """
    Load the results of a simulation
    :param simname: name of the simulation
    :return: final_state, t, e, m
    """

    simdir = "./simulations/{0}/".format(simname)

    final_state = np.load(simdir + "state.npy")
    results = np.load(simdir + "results.npy")

    e = results[:, 0]
    m = results[:, 1:4]

    # Build steps axis
    snapshots_params = np.load(simdir + "snapshots_params.npy")
    t = snapshots_params[:, 0]
    J = snapshots_params[:, 1]
    h = snapshots_params[:, 2]
    T = snapshots_params[:, 3]

    return final_state, t, J, h, T, e, m


def load_snapshots(simname):
    simdir = "./simulations/{0}/".format(simname)
    snapshots = np.load(simdir + "snapshots.npy")
    return snapshots


def load_set_results(setname):
    filelist = os.listdir(WORKING_DIRECTORY + setname + "/")
    simlist = []

    for f in filelist:
        if f.find(setname) >= 0:
            simlist.append(f)

    simlist.sort()
    simnumber = len(simlist)

    for i in range(0, simnumber):
        final_state_loaded, t_loaded, J_loaded, h_loaded, T_loaded, e_loaded, m_loaded = load_results(
            setname + "/" + simlist[i])
        if i == 0:
            final_state = np.zeros(shape=((simnumber,) + final_state_loaded.shape))
            t = np.zeros(shape=((simnumber,) + t_loaded.shape))
            J = np.zeros(shape=((simnumber,) + J_loaded.shape))
            h = np.zeros(shape=((simnumber,) + h_loaded.shape))
            T = np.zeros(shape=((simnumber,) + T_loaded.shape))
            e = np.zeros(shape=((simnumber,) + e_loaded.shape))
            m = np.zeros(shape=((simnumber,) + m_loaded.shape))

        final_state[i] = final_state_loaded
        t[i] = t_loaded
        J[i] = J_loaded
        h[i] = h_loaded
        T[i] = T_loaded
        e[i] = e_loaded
        m[i] = m_loaded

    return final_state, t, J, h, T, e, m


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
        sys = HeisenbergSystem(snapshots[i, :, :, :, :], J=J, h=h, T=T)
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


def compute_set_statistics(e, m):
    set_size = e.shape[0]
    e_mean = np.zeros(shape=set_size)
    e_rmsd = np.zeros(shape=set_size)
    m_mean = np.zeros(shape=(set_size, 3))
    m_rmsd = np.zeros(shape=(set_size, 3))

    for i in range(0, set_size):
        stats = compute_statistics(e[i], m[i])
        e_mean[i] = stats["e_mean"]
        e_rmsd[i] = stats["e_rmsd"]
        m_mean[i] = stats["m_mean"]
        m_rmsd[i] = stats["m_rmsd"]

    return e_mean, e_rmsd, m_mean, m_rmsd

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


def plot_state(snapshot):
    """
    Plot system state
    """
    nx = snapshot.shape[0]
    ny = snapshot.shape[1]
    nz = snapshot.shape[2]

    x, y, z = np.meshgrid(np.arange(0, nx),
                          np.arange(0, ny),
                          np.arange(0, nz))

    u = np.zeros(shape=(nx, ny, nz))
    v = np.zeros(shape=(nx, ny, nz))
    w = np.zeros(shape=(nx, ny, nz))

    for i, j, k in np.ndindex(nx, ny, nz):
        u[i, j, k], v[i, j, k], w[i, j, k] = sph2xyz(snapshot[i, j, k, 0], snapshot[i, j, k, 1])

    fig = plt.figure()
    ax: Axes3D = fig.gca(projection='3d')
    ax.quiver(x, y, z, u, v, w, pivot='middle')
    plt.show()
