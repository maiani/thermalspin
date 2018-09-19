#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Heisnberg model Monte Carlo simulator
"""

import numpy as np

from heisenberg_system import HeisenbergSystem


class HeisenbergSimulation:
    """
    Handler of the HeisenbergSystem simulation.
    It run the simulation and collect the results.
    """

    def __init__(self, nsteps, hsys: HeisenbergSystem, snapshot_delta):
        """
        :param nsteps: number of steps to be computed
        :param hsys: system to be evolved
        :param snapshot_delta: number of steps before taking a snapshot of the system
        """

        self.nsteps = nsteps
        self.system = hsys
        self.snapshot_delta = snapshot_delta

        self.snapshot_number = int(np.ceil(nsteps / snapshot_delta))
        self.snapshots_t = np.arange(0, nsteps, snapshot_delta)
        self.snapshots = np.zeros(shape=(self.snapshot_number, self.system.Nx, self.system.Ny, self.system.Nz, 2))

        self.snapshots[0:, :, :, :, :] = hsys.S.copy()

    def run(self):

        for t in range(0, self.nsteps):
            self.system.step()
            i = np.argmax(t == self.snapshots_t)
            if i != 0:
                self.snapshots[i, :, :, :, :] = self.system.S.copy()