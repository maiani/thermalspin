#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show simple data of the simulation
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

from skdylib.spherical_coordinates import sph2xyz, sph_dot
from thermalspin.heisenberg_system import site_energy

SIMULATIONS_DIRECTORY = "./simulations/"


# ----------------------------------------- LOADING --------------------------------------------------------------------

def load_results(simulation_name):
    """
    Load the results of a simulation
    :param simulation_name: name of the simulation
    :return: final_state, t, e, m
    """

    simdir = SIMULATIONS_DIRECTORY + f"{simulation_name}/"

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


def load_snapshots(simulation_name):
    """
    :param simulation_name: The name of the simulation
    :return: An array of the snapshots
    """
    simulation_directory = SIMULATIONS_DIRECTORY + f"{simulation_name}/"
    snapshots = np.load(simulation_directory + "snapshots.npy")
    return snapshots


def load_set_results(set_name, load_set_snapshots=False):
    filelist = os.listdir(SIMULATIONS_DIRECTORY + set_name + "/")
    simlist = []

    for f in filelist:
        simlist.append(f)

    simlist.sort()
    simnumber = len(simlist)
    snapshots = []

    for i in range(simnumber):
        try:
            final_state_loaded, t_loaded, J_loaded, h_loaded, T_loaded, e_loaded, m_loaded = load_results(
                set_name + "/" + simlist[i])

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
            L[i] = np.array(final_state_loaded.shape[0])
            t[i] = t_loaded
            J[i] = J_loaded
            h[i] = h_loaded
            T[i] = T_loaded
            e[i] = e_loaded
            m[i] = m_loaded

        except(Exception):
            print(f"Error in {simlist[i]}")

        if load_set_snapshots:
            snapshots.append(load_snapshots(set_name + "/" + simlist[i]))

    return final_state, L, t, J, h, T, e, m, snapshots


def arrange_set_results_LT(L_lst, t_lst, J_lst, h_lst, T_lst, e_lst, m_lst, final_state_lst, snapshots_lst=None):
    L_new = np.unique(L_lst)
    T_new = np.unique(T_lst)
    L_num = L_new.shape[0]
    T_num = T_new.shape[0]
    t_num = t_lst.shape[1]
    sim_num = t_lst.shape[0]

    final_state_new = [[None] * T_num for _ in range(L_num)]
    snapshots_new = [[None] * T_num for _ in range(L_num)]

    e_new = np.zeros(shape=(L_num, T_num, t_num))
    t_new = np.zeros(shape=(L_num, T_num, t_num))
    J_new = np.zeros(shape=(L_num, T_num, t_num))
    h_new = np.zeros(shape=(L_num, T_num, t_num))
    m_new = np.zeros(shape=(L_num, T_num, t_num, 3))

    for i in range(sim_num):
        T_idx = int(np.argmax(np.equal(T_new, T_lst[i, 0])))
        L_idx = int(np.argmax(np.equal(L_new, L_lst[i, 0])))

        final_state_new[L_idx][T_idx] = final_state_lst[i]
        if snapshots_lst is not None:
            snapshots_new[L_idx][T_idx] = snapshots_lst[i]

        e_new[L_idx, T_idx] = e_lst[i]
        t_new[L_idx, T_idx] = t_lst[i]
        J_new[L_idx, T_idx] = J_lst[i]
        h_new[L_idx, T_idx] = h_lst[i]
        m_new[L_idx, T_idx] = m_lst[i]

    return L_new, T_new, t_new, J_new, h_new, e_new, m_new, final_state_new, snapshots_new


def arrange_set_results_LH(L_lst, t_lst, J_lst, H_lst, T_lst, e_lst, m_lst, final_state_lst, snapshots=None):
    L_new = np.unique(L_lst)
    H_new = np.unique(H_lst)

    L_num = L_new.shape[0]
    H_num = H_new.shape[0]
    t_num = t_lst.shape[1]
    sim_num = t_lst.shape[0]

    final_state_new = [[None] * H_num for _ in range(L_num)]
    snapshots_new = [[None] * H_num for _ in range(L_num)]

    e_new = np.zeros(shape=(L_num, H_num, t_num))
    t_new = np.zeros(shape=(L_num, H_num, t_num))
    J_new = np.zeros(shape=(L_num, H_num, t_num))
    T_new = np.zeros(shape=(L_num, H_num, t_num))
    m_new = np.zeros(shape=(L_num, H_num, t_num, 3))

    for i in range(sim_num):
        Hz_idx = int(np.argmax(np.equal(H_new, H_lst[i, 0])))
        L_idx = int(np.argmax(np.equal(L_new, L_lst[i, 0])))

        final_state_new[L_idx][Hz_idx] = final_state_lst[i]
        if snapshots is not None:
            snapshots_new[L_idx][Hz_idx] = snapshots[i]

        e_new[L_idx, Hz_idx] = e_lst[i]
        t_new[L_idx, Hz_idx] = t_lst[i]
        J_new[L_idx, Hz_idx] = J_lst[i]
        T_new[L_idx, Hz_idx] = T_lst[i]
        m_new[L_idx, Hz_idx] = m_lst[i]

    return L_new, H_new, t_new, J_new, T_new, e_new, m_new, final_state_new, snapshots_new


# --------------------------------------------- COMPUTING --------------------------------------------------------------

@jit(nopython=True, cache=True)
def compute_site_energy(J, Hz, snapshots):
    t, nx, ny, nz, _ = snapshots.shape

    energy_per_site = np.zeros(shape=(t, nx, ny, nz))

    for n, i, j, k in np.ndindex(t, nx, ny, nz):
        energy_per_site[n, i, j, k] = site_energy(i, j, k, snapshots[n], J[n], Hz[n])

    return energy_per_site



def bootstrap(initial_samples, n, new_samples_number):
    old_samples_number = initial_samples.shape[0]
    new_shape = (new_samples_number, n)
    new_samples = np.zeros(shape=new_shape)

    for i in range(new_samples_number):
        indices = np.random.choice(old_samples_number, n)
        new_samples[i] = initial_samples.take(indices)

    return new_samples


@jit(nopython=True, cache=True)
def snapshot_sph2xyz(snapshot_sph):
    nx, ny, nz, u = snapshot_sph.shape
    snapshot_xyz = np.zeros(shape=(nx, ny, nz, 3))

    for i, j, k in np.ndindex(nx, ny, nz):
        snapshot_xyz[i, j, k] = sph2xyz(snapshot_sph[i, j, k, 0], snapshot_sph[i, j, k, 1])

    return snapshot_xyz


@jit(nopython=True, cache=True)
def snapshot_dot(snapshot1, snapshot2):
    nx, ny, nz, u = snapshot1.shape
    ret = np.zeros(shape=(nx, ny, nz))

    for i, j, k in np.ndindex(nx, ny, nz):
        ret[i, j, k] = sph_dot(snapshot1[i, j, k, 0], snapshot2[i, j, k, 0],
                               snapshot1[i, j, k, 1] - snapshot2[i, j, k, 1])
    return ret


# @jit(nopython=True, cache=True)
def time_correlation(snapshot1, snapshot2):
    """
    Compute the time correlation between two snapshots (averaging on each site)
    """
    s1 = np.mean(snapshot_sph2xyz(snapshot1), axis=(0, 1, 2))
    s2 = np.mean(snapshot_sph2xyz(snapshot2), axis=(0, 1, 2))
    s1s2 = s1.dot(s2)
    return np.mean(snapshot_dot(snapshot1, snapshot2), axis=(0, 1, 2)) - s1s2


@jit(nopython=True, cache=True)
def translate_snapshot(snapshot, x, y, z):
    nx, ny, nz, u = snapshot.shape
    ret = np.zeros(shape=snapshot.shape)

    for i, j, k in np.ndindex(nx, ny, nz):
        ret[i, j, k] = snapshot[(i + x) % nx, (j + y) % ny, (k + z) % nz]

    return ret


# @jit(nopython=True, cache=True)
def spatial_correlation_matrix(snapshot):
    nx, ny, nz, u = snapshot.shape
    ret = np.zeros(shape=(nx, ny, nz))

    s = np.mean(snapshot_sph2xyz(snapshot), axis=(0, 1, 2))
    s1s2 = s.dot(s)

    for i, j, k in np.ndindex(nx, ny, nz):
        ret[i, j, k] = np.mean(snapshot_dot(snapshot, translate_snapshot(snapshot, i, j, k)), axis=(0, 1, 2)) - s1s2
    return ret


# @jit(nopython=True, cache=True)
def radial_distribution(correlation_matrix):
    nx, ny, nz = correlation_matrix.shape
    corr = np.zeros(shape=(nx * ny * nz, 2)) * np.NaN

    l = 0
    for i, j, k in np.ndindex(nx, ny, nz):
        corr[l, 0] = np.sqrt(i ** 2 + j ** 2 + k ** 2)
        corr[l, 1] = correlation_matrix[i, j, k]
        l += 1

    r = np.unique(corr[:, 0])
    c = np.zeros(shape=r.shape)

    for i in range(r.shape[0]):
        c[i] = np.mean(corr[(corr[:] == r[i])[:, 0], 1])

    return r, c


# ---------------------------------------------- PLOTTING --------------------------------------------------------------


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

    c = np.zeros(shape=(nx, ny, nz, 4))
    c[:, :, :, 0] = u
    c[:, :, :, 1] = v
    c[:, :, :, 2] = w
    c[:, :, :, 3] = np.ones(shape=(nx, ny, nz))
    c = np.abs(c)

    c2 = np.zeros(shape=(nx * ny * nz, 4))
    l = 0
    for i, j, k in np.ndindex((nx, ny, nz)):
        c2[l] = c[i, j, k]
        l += 1

    c3 = np.concatenate((c2, np.repeat(c2, 2, axis=0)), axis=0)

    fig = plt.figure()
    ax: Axes3D = fig.gca(projection='3d')
    ax.quiver(x, y, z, u, v, w, pivot='middle', color=c3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def plot_state_2D(snapshot):
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

    c = np.zeros(shape=(nx, ny, nz, 4))
    c[:, :, :, 0] = u
    c[:, :, :, 1] = v
    c[:, :, :, 2] = w
    c[:, :, :, 3] = np.ones(shape=(nx, ny, nz))
    c = np.abs(c)

    c2 = np.zeros(shape=(nx * ny * nz, 4))
    l = 0
    for i, j, k in np.ndindex((nx, ny, nz)):
        c2[l] = c[i, j, k]
        l += 1

    c3 = np.concatenate((c2, np.repeat(c2, 2, axis=0)), axis=0)

    fig = plt.figure()
    ax: Axes3D = fig.gca(projection='3d')
    ax.quiver(x, y, z, u, v, w, pivot='middle', color=c3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_zlim([-1.1, 1.1])

    plt.show()


def plot_spin_directions(snapshot):
    """
    Plot spherical representation of spins
    """
    nx = snapshot.shape[0]
    ny = snapshot.shape[1]
    nz = snapshot.shape[2]

    points = np.zeros(shape=(nx * ny * nz, 3))

    n = 0
    for i, j, k in np.ndindex(nx, ny, nz):
        points[n] = sph2xyz(snapshot[i, j, k, 0], snapshot[i, j, k, 1])
        n += 1

    fig = plt.figure()
    ax: Axes3D = fig.gca(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.abs(points), s=2)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
