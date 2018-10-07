#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Heisenberg model Monte Carlo simulator
"""

import json
import os
import shutil
import time

import numpy as np

from heisenberg_system import HeisenbergSystem
from kent_distribution.kent_distribution import kent2
from math_utils import sph_u_rand, xyz2sph

SNAPSHOTS_ARRAY_DIMENSION = int(5e4)


class HeisenbergSimulation:
    """
    Handler of the HeisenbergSystem simulation.
    It run the simulation and collect the results.
    """

    def __init__(self, hsys: HeisenbergSystem, take_states_snapshots=False):
        """
        :param hsys: system to be evolved
        """

        self.system = hsys
        self.steps_counter = 0
        self.snapshots_counter = 0

        if take_states_snapshots:
            self.snapshots = np.zeros(
                shape=(SNAPSHOTS_ARRAY_DIMENSION, self.system.nx, self.system.ny, self.system.nz, 2))
        else:
            self.snapshots = None

        self.snapshots_t = np.zeros(shape=SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_e = np.zeros(shape=SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_m = np.zeros(shape=(SNAPSHOTS_ARRAY_DIMENSION, 3))

        self.snapshots_J = np.zeros(shape=SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_T = np.zeros(shape=SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_Hz = np.zeros(shape=SNAPSHOTS_ARRAY_DIMENSION)

        self.take_snapshot()

    def run(self, nsteps):
        """
        Evolve the system for a given number of steps
        :param nsteps: The number of steps
        """
        self.steps_counter += nsteps
        for t in range(1, nsteps + 1):
            self.system.step()

    def take_snapshot(self):
        """"
        Take a snapshot of the system, parameters and results
        """
        # TODO: Fix for snapshots exceeding the array dimension

        if self.snapshots is not None:
            self.snapshots[self.snapshots_counter, :, :, :, :] = self.system.state.copy()

        self.snapshots_t[self.snapshots_counter] = self.steps_counter
        self.snapshots_e[self.snapshots_counter] = self.system.energy
        self.snapshots_m[self.snapshots_counter, :] = self.system.magnetization

        self.snapshots_J[self.snapshots_counter] = self.system.J
        self.snapshots_T[self.snapshots_counter] = self.system.T
        self.snapshots_Hz[self.snapshots_counter] = self.system.Hz

        self.snapshots_counter += 1

    def run_with_snapshots(self, steps_number, delta_snapshots, verbose=False):
        """
        Evolve the system while taking snapshots
        :param steps_number: Number of steps to be computed
        :param delta_snapshots: Distance between snapshots
        """

        if steps_number % delta_snapshots != 0:
            raise Exception("steps_number must be multiple of delta_snapshots")

        nsnapshots = int(steps_number / delta_snapshots)
        for t in range(0, nsnapshots):
            self.run(delta_snapshots)
            self.take_snapshot()
            if verbose:
                print(f"Step number {self.steps_counter}", end="\r")


# Functions for initialization and saving to disk the results of a simulation

def init_simulation_aligned(simdir, nx, ny, nz, params, theta_0=None, phi_0=None):
    """
    Generate a lattice of spins aligned toward an axis
    :param simdir: Directory of the simulation
    :param nx: Number of x cells
    :param ny: Number of y cells
    :param nz: Number of z cells
    :param params: parameters of the simulation
    :param phi_0:
    :param theta_0:
    """
    shutil.rmtree(simdir, ignore_errors=True)

    state = np.ones(shape=(nx, ny, nz, 2))
    state[:, :, :, 0] = state[:, :, :, 0] * theta_0
    state[:, :, :, 1] = state[:, :, :, 1] * phi_0

    os.makedirs(simdir)
    params_file = open(simdir + "params.json", "w")
    json.dump(params, params_file, indent=2)
    np.save(simdir + "state.npy", state)


def init_simulation_tilted(simdir, nx, ny, nz, params):
    """
    Generate a lattice of spins aligned toward tan axis if specified, random if not
    :param simdir: Directory of the simulation
    :param nx: Number of x cells
    :param ny: Number of y cells
    :param nz: Number of z cells
    :param params: parameters of the simulation
    """

    shutil.rmtree(simdir, ignore_errors=True)

    state = np.ones(shape=(nx, ny, nz, 2))

    gamma1 = np.array([0, 0, 1], dtype=np.float)
    gamma2 = np.array([0, 1, 0], dtype=np.float)
    gamma3 = np.array([1, 0, 0], dtype=np.float)
    kent = kent2(gamma1, gamma2, gamma3, kappa=20, beta=0)

    for i, j, k in np.ndindex((nx, ny, nz)):
        state[i, j, k, :] = xyz2sph(kent.rvs())

    os.makedirs(simdir)
    params_file = open(simdir + "params.json", "w")
    json.dump(params, params_file, indent=2)
    np.save(simdir + "state.npy", state)


def init_simulation_random(simdir, nx, ny, nz, params):
    """
    Generate a lattice of spins aligned toward tan axis if specified, random if not
    :param simdir: Directory of the simulation
    :param nx: Number of x cells
    :param ny: Number of y cells
    :param nz: Number of z cells
    :param params: parameters of the simulation
    """
    shutil.rmtree(simdir, ignore_errors=True)

    state = np.zeros(shape=(nx, ny, nz, 2))
    for i, j, k in np.ndindex(nx, ny, nz):
        theta_r, phi_r = sph_u_rand()
        state[i, j, k, 0] = theta_r
        state[i, j, k, 1] = phi_r

    os.makedirs(simdir)
    params_file = open(simdir + "params.json", "w")
    json.dump(params, params_file, indent=2)
    np.save(simdir + "state.npy", state)


def run_simulation(simdir, save_snapshots=False, verbose=True):
    """
    Run a simulation and save to disk the results
    :param simdir: the directory of the simulation
    :param save_snapshots: boolean for saving also the snapshots' states
    """

    if os.path.isfile(simdir + "params.json"):
        params_file = open(simdir + "params.json", "r")
        params = json.load(params_file)
    else:
        raise Exception("Missing params.json file")

    if os.path.isfile(simdir + "state.npy"):
        state = np.load(simdir + "state.npy")
    else:
        raise Exception("Missing state.npy file")

    param_J = np.array(params["param_J"])
    param_Hz = np.array(params["param_Hz"])
    param_T = np.array(params["param_T"])
    steps_number = params["steps_number"]
    delta_snapshots = params["delta_snapshots"]

    sys = HeisenbergSystem(state, param_J[0], param_Hz[0], param_T[0])
    hsim = HeisenbergSimulation(sys, take_states_snapshots=save_snapshots)

    for i in range(param_T.shape[0]):
        print(f"Simulation stage:   {i}\n"
              f"Temperature:        {param_T[i]}\n"
              f"Hz:                 {param_Hz[i]}\n"
              f"Steps number:       {steps_number}\n"
              f"Delta snapshots:    {delta_snapshots}\n")

        hsim.system.J = param_J[i]
        hsim.system.T = param_T[i]
        hsim.system.Hz = param_Hz[i]
        start_time = time.time()
        hsim.run_with_snapshots(steps_number, delta_snapshots, verbose=verbose)
        end_time = time.time()
        run_time = end_time - start_time

        print(f"Stage completed in {run_time} seconds\n")

    print("Saving results ...", end="")
    start = time.time()

    # Save the last state
    np.save(simdir + "state.npy", hsim.system.state)

    # Collect the results of the simulation
    new_results = np.zeros(shape=(hsim.snapshots_counter, 4))
    new_results[:, 0] = hsim.snapshots_e[:hsim.snapshots_counter]
    new_results[:, 1:4] = hsim.snapshots_m[:hsim.snapshots_counter]

    # Collect the snapshots and params
    new_snapshots_params = np.zeros(shape=(hsim.snapshots_counter, 4))
    new_snapshots_params[:, 0] = hsim.snapshots_t[:hsim.snapshots_counter]
    new_snapshots_params[:, 1] = hsim.snapshots_J[:hsim.snapshots_counter]
    new_snapshots_params[:, 2] = hsim.snapshots_Hz[:hsim.snapshots_counter]
    new_snapshots_params[:, 3] = hsim.snapshots_T[:hsim.snapshots_counter]

    # If old data is found, append the new one
    if os.path.isfile(simdir + "snapshots_params.npy") and os.path.isfile(simdir + "results.npy"):

        old_results = np.load(simdir + "results.npy")
        results = np.concatenate((old_results, new_results[1:]))

        old_snapshots_params = np.load(simdir + "snapshots_params.npy")
        last_t = old_snapshots_params[-1, 0]
        new_snapshots_params[:, 0] += last_t
        snapshots_params = np.concatenate((old_snapshots_params, new_snapshots_params[1:]))

    else:
        snapshots_params = new_snapshots_params
        results = new_results

    # Save all
    np.save(simdir + "snapshots_params.npy", snapshots_params)
    np.save(simdir + "results.npy", results)

    if save_snapshots:
        new_snapshots = hsim.snapshots[:hsim.snapshots_counter]
        if os.path.isfile(simdir + "snapshots.npy"):
            old_snapshots = np.load(simdir + "snapshots.npy")
            snapshots = np.concatenate((old_snapshots, new_snapshots[1:]))

        else:
            snapshots = new_snapshots
        np.save(simdir + "snapshots.npy", snapshots)

    end = time.time()
    saving_time = end - start
    print("done in {0} seconds.".format(saving_time))
