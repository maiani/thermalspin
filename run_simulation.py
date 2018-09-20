#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a simulation with default parameters
"""
import json
import os
import time

import numpy as np

from heisenberg_simulation import HeisenbergSimulation
from heisenberg_system import HeisenbergSystem

default_params = {"J": 1, "h": 0, "T": 1, "nsteps": 20000, "delta_snp": 50}


def run_simulation(simname):
    simdir = "./simulations/{0}/".format(simname)
    os.makedirs(simdir, exist_ok=True)

    # If a param.json file is found, load it, else create one with default parameters
    if os.path.isfile(simdir + "params.json"):
        params_file = open(simdir + "params.json", "r")
        params = json.load(params_file)
        delta_snp = params["delta_snp"]
    else:
        params = default_params
        params_file = open(simdir + "params.json", "w")
        json.dump(params, params_file, sort_keys=True, indent=4)

    J = params["J"]
    h = params["h"]
    T = params["T"]
    nsteps = params["nsteps"]
    delta_snp = params["delta_snp"]

    sys = HeisenbergSystem(J=J, h=h, T=T)

    Nx = 16
    Ny = 16
    Nz = 1
    sys.build_aligned_system(Nx, Ny, Nz)

    hsim = HeisenbergSimulation(nsteps, sys, delta_snp)
    start = time.time()
    hsim.run()
    end = time.time()
    run_time = end - start
    print("Simulation completed in {0} seconds".format(run_time))

    np.save(simdir + "snapshots.npy", hsim.snapshots)


if __name__ == "__main__":
    run_simulation("sim_0")
