#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run multiple instance of Heisenberg
"""

import getopt
import os
import sys

import numpy as np

from heisenberg_simulation import init_simulation, run_simulation

WORKING_DIRECTORY = "./simulations/"
DEFAULT_PARAMS = dict(J=1, h=0, T=0.5, nsteps=5e7, delta_snp=10000)


def init_temperature_set(setname, T_a, T_b, dT, nx, ny, nz, theta_0=None, phi_0=None, ):
    T_set = np.arange(T_a, T_b + 0.01, dT)

    for T_i in T_set:
        simdir = WORKING_DIRECTORY + setname + "/" + setname + "_T" + str(T_i) + "/"
        params = DEFAULT_PARAMS
        params["T"] = T_i
        init_simulation(simdir, nx, ny, nz, params=params, theta_0=theta_0, phi_0=phi_0)


def run_set(setname, save_snapshots=False):
    filelist = os.listdir(WORKING_DIRECTORY + setname + "/")
    simlist = []

    for filename in filelist:
        if filename.find(setname) >= 0:
            simlist.append(filename)

    for simulation in simlist:
        run_simulation(WORKING_DIRECTORY + setname + "/" + simulation + "/", save_snapshots=save_snapshots)


def usage():
    print("""
    Usage: heisenberg_set.py [OPTIONS] [PARAMETERS]\n
    -i, --init=SETNAME                Initialize a set of simulations, need to specify next a dimension and a magnetization
    -r, --run=SETNAME                 Run a set of simulations named SETNAME
    -d, --dimensions=SIZE             Generate a default simulation with SIZE specified e.g. 10x10x10
    -m, --magnetization=DIRECTION     Initial magnetization along DIRECTION specified like 0,0
    -t, --temperature=TEMP            Specify the range of temperature with TEMP like T_initial,T_final,dT e.g 0.5,3.5,1  
    -h, --help                        Shows this message
    """)


if __name__ == "__main__":
    mode = None
    setname = None
    nx, ny, nz = (None, None, None)
    theta_0, phi_0 = (None, None)
    Ti, Tf, dt = (None, None, None)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:i:d:m:t:",
                                   ["help", "initialize=", "run=", "dimensions=", "temperatures="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-r", "--run"):
            mode = "run"
            setname = arg
        elif opt in ("-i", "--initialize"):
            mode = "init"
            setname = arg
        elif opt in ("-d", "--dimensions"):
            nx, ny, nz = arg.split("x")
        elif opt in ("-m", "--magnetization"):
            theta_0, phi_0 = arg.split(",")
        elif opt in ("-t", "--temperature"):
            Ti, Tf, dT = arg.split(",")

    if mode == "run":
        print(f"Running simulation in set {setname}")
        run_set(setname)
    elif mode == "init":
        if theta_0 is None:
            init_temperature_set(setname, float(Ti), float(Tf), float(dT), int(nx), int(ny), int(nz))
        else:
            init_temperature_set(setname, float(Ti), float(Tf), float(dT), int(nx), int(ny), int(nz), theta_0, phi_0)
    else:
        usage()
        sys.exit(2)

    print("Finished")
