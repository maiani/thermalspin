#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a simulation with default parameters
"""

import getopt
import json
import os
import shutil
import sys
import time

import numpy as np

from heisenberg_simulation import HeisenbergSimulation
from heisenberg_system import HeisenbergSystem
from math_utils import rand_sph


def init_simulation(simname, nx, ny, nz, params=None, theta_0=None, phi_0=None):
    """
    Generate a lattice of spins aligned upward z
    :param simname: Name of the simulation
    :param nx: Number of x cells
    :param ny: Number of y cells
    :param nz: Number of z cells
    :param params: parameters of the simulation
    :param phi_0:
    :param theta_0:
    """
    default_params = dict(J=1, h=0, T=1, nsteps=5000000, delta_snp=2500)

    if not params:
        params = default_params

    simdir = "./simulations/" + simname + "/"
    shutil.rmtree(simdir, ignore_errors=True)

    if theta_0 is None:
        state = np.zeros(shape=(nx, ny, nz, 2))
        for i, j, k in np.ndindex(nx, ny, nz):
            theta_r, phi_r = rand_sph()
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


def run_simulation(simname):
    simdir = "./simulations/{0}/".format(simname)

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

    hsim = HeisenbergSimulation(nsteps, sys, delta_snp)
    start = time.time()
    hsim.run()
    end = time.time()
    run_time = end - start
    print("Simulation completed in {0} seconds".format(run_time))


    print("Saving results ...", end="")

    results = np.zeros(shape=(hsim.snapshot_number, 4))
    results[:, 0] = hsim.snapshots_e
    results[:, 1:4] = hsim.snapshots_m

    if os.path.isfile(simdir + "snapshots.npy") and os.path.isfile(simdir + "snapshots_t.npy") and os.path.isfile(
            simdir + "results.npy"):
        old_snapshots = np.load(simdir + "snapshots.npy")
        snapshots = np.concatenate((old_snapshots, hsim.snapshots[1:]))

        old_results = np.load(simdir + "results.npy")
        results = np.concatenate((old_results, results[1:]))

        old_snapshots_t = np.load(simdir + "snapshots_t.npy")
        last_t = old_snapshots_t[-1]
        new_snapshots_t = hsim.snapshots_t + np.ones(hsim.snapshots_t.shape) * last_t
        snapshots_t = np.concatenate((old_snapshots_t, new_snapshots_t[1:]))
    else:
        snapshots = hsim.snapshots
        snapshots_t = hsim.snapshots_t

    np.save(simdir + "snapshots.npy", snapshots)
    np.save(simdir + "snapshots_t.npy", snapshots_t)
    np.save(simdir + "state.npy", hsim.system.state)
    np.save(simdir + "results.npy", results)
    print("done")

def usage():
    print("""
    Usage: heisenberg.py [OPTIONS] [PARAMETERS]\n
    -r, --run=SIMNAME                 Run a simulation named SIMNAME
    -d, --dimensions=SIZE             Generate a default simulation with SIZE specified e.g. 10x10x10
    -m, --magnetization=DIRECTION     Initial magnetization along DIRECTION specified like 0,0
    -h, --help                        Shows this message
    """)


if __name__ == "__main__":
    # Fallback
    mode = ""
    simname = "default"
    nx, ny, nz = (8, 8, 8)
    theta_0, phi_0 = (None, None)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:i:d:m:", ["help", "initialize=", "run=", "dimensions="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-r", "--run"):
            mode = "run"
            simname = arg
        elif opt in ("-i", "--initialize"):
            mode = "init"
            simname = arg
        elif opt in ("-d", "--dimensions"):
            nx, ny, nz = arg.split("x")
        elif opt in ("-m", "--magnetization"):
            theta_0, phi_0 = arg.split(",")

    if mode == "run":
        print(f"Running simulation {simname}")
        run_simulation(simname)
    elif mode == "init":
        if theta_0 is None:
            init_simulation(simname, int(nx), int(ny), int(nz))
            print(f"Default simulation {simname} generated withe default params. \n"
                  f"Lattice has dimensions {nx}x{ny}x{nz} \n"
                  f"Random initial magnetization")
        else:
            init_simulation(simname, int(nx), int(ny), int(nz), theta_0=int(theta_0), phi_0=int(phi_0))
            print(f"Default simulation {simname} generated withe default params. \n"
                  f"Lattice has dimensions {nx}x{ny}x{nz} \n"
                  f"Initial magnetization ({theta_0},{phi_0})")
    else:
        usage()
        sys.exit(2)

    print("Finished")
