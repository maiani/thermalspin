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


def run_test_simulation():
    """
    Run a test simulation with predefined parameters
    """
    np.random.seed(1)
    test_params = dict(J=25, h=0, T=5, nsteps=2000000, delta_snp=500)
    nx, ny, nz = (8, 8, 8)
    init_simulation("test", nx, ny, nz, params=test_params)
    run_simulation("test")


def init_simulation(simname, nx, ny, nz, params=None):
    """
    Generate a lattice of spins aligned upward z
    :param nx: Number of x cells
    :param ny: Number of y cells
    :param nz: Number of z cells
    :param params: parameters of the simulation
    """
    default_params = dict(J=10, h=0, T=1, nsteps=100000, delta_snp=1000)

    if not params:
        params = default_params

    simdir = "./simulations/" + simname + "/"
    shutil.rmtree(simdir, ignore_errors=True)
    state = np.zeros(shape=(nx, ny, nz, 2))

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
    if os.path.isfile(simdir + "snapshots.npy") and os.path.isfile(simdir + "snapshots_t.npy"):
        old_snapshots = np.load(simdir + "snapshots.npy")
        snapshots = np.concatenate((old_snapshots, hsim.snapshots[1:]))

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
    print("done")

def usage():
    print("""
    Usage: heisenberg.py [OPTIONS] [PARAMETERS]\n
    -r, --run=SIMNAME                 Run a simulation named SIMNAME
    -d, --default=SIZE                Generate a default simulation with SIZE specified e.g. 10x10x10
    -t, --test                        Run a test simulations with default parameters to measure performances
    -h, --help                        Shows this message
    """)


if __name__ == "__main__":
    mode = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "htr:i:d:", ["help", "test", "initialize=", "run=", "dimensions="])
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
        elif opt in ("-t", "--test"):
            mode = "test"

    if mode == "run":
        print(f"Running simulation {simname}")
        run_simulation(simname)
    elif mode == "init":
        init_simulation(simname, int(nx), int(ny), int(nz))
        print(f"Default simulation {simname} generated withe default params. \nLattice has dimensions {nx}x{ny}x{nz}")
    elif mode == "test":
        run_test_simulation()
    else:
        usage()
        sys.exit(2)

    print("Finished")
