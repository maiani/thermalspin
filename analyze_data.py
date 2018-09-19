#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show simple data of the simulation
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from heisenberg_system import HeisenbergSystem

params_file = open('./output/sim1/params.json', 'r')
params = json.load(params_file)
J = params["J"]
h = params["h"]
T = params["T"]
delta = params["delta"]

snapshots = np.load("./output/sim1/data.npy")
snapshots_number = snapshots.shape[0]
Nx = snapshots.shape[1]
Ny = snapshots.shape[2]
Nz = snapshots.shape[3]

# Build steps axis
t = np.arange(0, snapshots_number) * delta

# Compute energy and magnetization
E = np.zeros(shape=(snapshots_number))
M = np.zeros(shape=(snapshots_number, 3))
for i in np.arange(0, snapshots_number):
    sys = HeisenbergSystem(snapshots[i, :, :, :, :], J=J, h=h, T=T)
    E[i] = sys.H
    M[i, :] = sys.M

plt.figure()
plt.plot(t, E)
plt.title("Energy")

plt.figure()
plt.plot(t, M[:, 2])
plt.title("Mz")
