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
from math_utils import sph_u_rand


class HeisenbergSimulation:
    """
    Handler of the HeisenbergSystem simulation.
    It run the simulation and collect the results.
    """

    def __init__(self, hsys: HeisenbergSystem, take_states_snapshots=False):
        """
        :param hsys: system to be evolved
        """

        self.SNAPSHOTS_ARRAY_DIMENSION = int(10e3)

        self.system = hsys
        self.steps_counter = 0
        self.snapshots_counter = 0

        if take_states_snapshots:
            self.snapshots = np.zeros(
                shape=(self.SNAPSHOTS_ARRAY_DIMENSION, self.system.nx, self.system.ny, self.system.nz, 2))
        else:
            self.snapshots = None

        self.snapshots_t = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_e = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_m = np.zeros(shape=(self.SNAPSHOTS_ARRAY_DIMENSION, 3))

        self.snapshots_J = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_T = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_h = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)

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
        print(f"Step number: {self.steps_counter}")

        if self.snapshots is not None:
            self.snapshots[self.snapshots_counter, :, :, :, :] = self.system.state.copy()

        self.snapshots_t[self.snapshots_counter] = self.steps_counter
        self.snapshots_e[self.snapshots_counter] = self.system.energy
        self.snapshots_m[self.snapshots_counter, :] = self.system.magnetization

        self.snapshots_J[self.snapshots_counter] = self.system.J
        self.snapshots_T[self.snapshots_counter] = self.system.T
        self.snapshots_h[self.snapshots_counter] = self.system.h

        self.snapshots_counter += 1

    def run_with_snapshots(self, nstep, delta_snapshots):
        """
        Evolve the system while taking snapshots
        :param nstep: Number of steps to be computed
        :param delta_snapshots: Distance between snapshots
        """

        if nstep % delta_snapshots != 0:
            raise Exception("nstep must be multiple of delta_snapshots")

        nsnapshots = int(nstep / delta_snapshots)
        for t in range(0, nsnapshots):
            self.run(delta_snapshots)
            self.take_snapshot()


# Functions for initialization and saving to disk the results of a simulation

DEFAULT_PARAMS = dict(J=1, h=0, T=0.5, nsteps=int(5e5), delta_snp=int(1e4))


def init_simulation(simdir, nx, ny, nz, params=None, theta_0=None, phi_0=None):
    """
    Generate a lattice of spins aligned toward tan axis if specified, random if not
    :param simdir: Directory of the simulation
    :param nx: Number of x cells
    :param ny: Number of y cells
    :param nz: Number of z cells
    :param params: parameters of the simulation
    :param phi_0:
    :param theta_0:
    """

    if not params:
        params = DEFAULT_PARAMS

    shutil.rmtree(simdir, ignore_errors=True)

    if theta_0 is None:
        state = np.zeros(shape=(nx, ny, nz, 2))
        for i, j, k in np.ndindex(nx, ny, nz):
            theta_r, phi_r = sph_u_rand()
            state[i, j, k, 0] = theta_r
            state[i, j, k, 1] = phi_r

    else:
        state = np.ones(shape=(nx, ny, nz, 2))
        state[:, :, :, 0] = state[:, :, :, 0] * theta_0
        state[:, :, :, 1] = state[:, :, :, 1] * phi_0

    os.makedirs(simdir)
    params_file = open(simdir + "params.json", "w")
    json.dump(params, params_file, sort_keys=True, indent=4)
    np.save(simdir + "state.npy", state)


def run_simulation(simdir, save_snapshots=False):
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

    J = params["J"]
    h = params["h"]
    T = params["T"]
    nsteps = params["nsteps"]
    delta_snp = params["delta_snp"]

    sys = HeisenbergSystem(state, J, h, T)

    hsim = HeisenbergSimulation(sys, take_states_snapshots=save_snapshots)
    start = time.time()
    hsim.run_with_snapshots(nsteps, delta_snp)
    end = time.time()
    run_time = end - start
    print("Simulation completed in {0} seconds".format(run_time))

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
    new_snapshots_params[:, 2] = hsim.snapshots_h[:hsim.snapshots_counter]
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
