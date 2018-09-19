#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a simulation with default parameters
"""
import json
import os

import numpy as np

from heisenberg_simulation import HeisenbergSimulation
from heisenberg_system import HeisenbergSystem

J = 1
h = 0
T = 1
sys = HeisenbergSystem(J=J, h=h, T=T)

Nx = 10
Ny = 10
Nz = 1
sys.build_aligned_system(Nx, Ny, Nz)

nsteps = 10000
delta = 10
hsim = HeisenbergSimulation(nsteps, sys, delta)
hsim.run()

os.makedirs("./output/sim1", exist_ok=True)

np.save("./output/sim1/data.npy", hsim.snapshots)

params = {"J": J, "h": h, "T": T, "delta": delta}
params_file = open('./output/sim1/params.json', 'w')
json.dump(params, params_file, sort_keys=True, indent=4)
