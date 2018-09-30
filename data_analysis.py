#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show simple data of the simulation
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from math_utils import sph2xyz

SIMULATIONS_DIRECTORY = "./simulations/"


def load_results(simname):
    """
    Load the results of a simulation
    :param simname: name of the simulation
    :return: final_state, t, e, m
    """

    simdir = SIMULATIONS_DIRECTORY + f"{simname}/"

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
    simdir = SIMULATIONS_DIRECTORY + f"{simname}/"
    snapshots = np.load(simdir + "snapshots.npy")
    return snapshots


def load_set_results(setname):
    filelist = os.listdir(SIMULATIONS_DIRECTORY + setname + "/")
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
            final_state = []
            L = np.zeros(shape=(simnumber, 3))
            t = np.zeros(shape=((simnumber,) + t_loaded.shape))
            J = np.zeros(shape=((simnumber,) + J_loaded.shape))
            h = np.zeros(shape=((simnumber,) + h_loaded.shape))
            T = np.zeros(shape=((simnumber,) + T_loaded.shape))
            e = np.zeros(shape=((simnumber,) + e_loaded.shape))
            m = np.zeros(shape=((simnumber,) + m_loaded.shape))

        final_state.append(final_state_loaded)
        L[i] = np.array(final_state_loaded[:, :, :, 0].shape)
        t[i] = t_loaded
        J[i] = J_loaded
        h[i] = h_loaded
        T[i] = T_loaded
        e[i] = e_loaded
        m[i] = m_loaded

    return final_state, L, t, J, h, T, e, m


def arrange_set_results(L_lst, t_lst, J_lst, h_lst, T_lst, e_lst, m_lst, final_state_lst):
    L_new = np.unique(L_lst)
    T_new = np.unique(T_lst)
    L_num = L_new.shape[0]
    T_num = T_new.shape[0]
    t_num = t_lst.shape[1]
    sim_num = t_lst.shape[0]

    tmp_array = [None] * T_num
    final_state_new = [tmp_array] * L_num

    e_new = np.zeros(shape=(L_num, T_num, t_num))
    t_new = np.zeros(shape=(L_num, T_num, t_num))
    J_new = np.zeros(shape=(L_num, T_num, t_num))
    h_new = np.zeros(shape=(L_num, T_num, t_num))
    m_new = np.zeros(shape=(L_num, T_num, t_num, 3))

    for i in range(sim_num):
        T_idx = int(np.argmax(np.equal(T_new, T_lst[i, 0])))
        L_idx = int(np.argmax(np.equal(L_new, L_lst[i, 0])))

        final_state_new[L_idx][T_idx] = final_state_lst[i]

        e_new[L_idx, T_idx] = e_lst[i]
        t_new[L_idx, T_idx] = t_lst[i]
        J_new[L_idx, T_idx] = J_lst[i]
        h_new[L_idx, T_idx] = h_lst[i]
        m_new[L_idx, T_idx] = m_lst[i]

    return L_new, T_new, t_new, J_new, h_new, e_new, m_new, final_state_new


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
