#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Heisenberg model Monte Carlo simulator
"""

import numpy as np

from heisenberg_system import HeisenbergSystem


class HeisenbergSimulation:
    """
    Handler of the HeisenbergSystem simulation.
    It run the simulation and collect the results.
    """

    def __init__(self, hsys: HeisenbergSystem):
        """
        :param hsys: system to be evolved
        """

        self.SNAPSHOTS_ARRAY_DIMENSION = int(10e3)

        self.system = hsys
        self.steps_counter = 0
        self.snapshots_counter = 0

        self.snapshots = np.zeros(
            shape=(self.SNAPSHOTS_ARRAY_DIMENSION, self.system.nx, self.system.ny, self.system.nz, 2))

        self.snapshots_t = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_e = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_m = np.zeros(shape=(self.SNAPSHOTS_ARRAY_DIMENSION, 3))

        self.snapshots_J = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_T = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)
        self.snapshots_h = np.zeros(shape=self.SNAPSHOTS_ARRAY_DIMENSION)

        self.take_snapshot()

    def run(self, nsteps):
        self.steps_counter += nsteps
        for t in range(1, nsteps + 1):
            self.system.step()

    def take_snapshot(self):
        # TODO: Fix for snapshots exceeding the array dimension
        print(f"Step number: {self.steps_counter}")
        self.snapshots[self.snapshots_counter, :, :, :, :] = self.system.state.copy()
        self.snapshots_t[self.snapshots_counter] = self.steps_counter
        self.snapshots_e[self.snapshots_counter] = self.system.energy
        self.snapshots_m[self.snapshots_counter, :] = self.system.magnetization

        self.snapshots_J[self.snapshots_counter] = self.system.J
        self.snapshots_T[self.snapshots_counter] = self.system.T
        self.snapshots_h[self.snapshots_counter] = self.system.h

        self.snapshots_counter += 1

    def run_with_snapshots(self, nstep, delta_snapshots):
        if nstep % delta_snapshots != 0:
            raise Exception("nstep must be multiple of delta_snapshots")

        nsnapshots = int(nstep / delta_snapshots)
        for t in range(0, nsnapshots):
            self.run(delta_snapshots)
            self.take_snapshot()
