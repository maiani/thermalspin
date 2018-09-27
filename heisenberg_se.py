#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
User interface for initializing and run simulations
"""

import getopt
import json
import os
import sys

from heisenberg_simulation import init_simulation, run_simulation

CONFIG_FILE_NAME = "config.json"

PROCESSES_NUMBER = None
WORKING_DIRECTORY = None
DEFAULT_PARAMS = None


def read_config_file():
    global PROCESSES_NUMBER
    global WORKING_DIRECTORY
    global DEFAULT_PARAMS

    if os.path.isfile(CONFIG_FILE_NAME):
        config_file = open(CONFIG_FILE_NAME, "r")
        config = json.load(config_file)
    else:
        raise Exception("Missing params.json file")

    PROCESSES_NUMBER = int(config["process_number"])
    WORKING_DIRECTORY = config["simulations_directory"]
    nsteps = int(config["steps_number"])
    delta_snp = int(config["delta_snapshots"])
    default_J = config["default_J"]
    default_h = config["default_h"]
    default_T = config["default_T"]

    DEFAULT_PARAMS = dict(J=default_J, h=default_h, T=default_T, nsteps=nsteps, delta_snp=delta_snp)


def usage():
    print("""
    Single ensemble simulation.
    Usage: heisenberg_se.py [OPTIONS] [PARAMETERS]\n
    -i, --init=SIMNAME                Initialize a simulation, need to specify next a dimension and a magnetization
    -r, --run=SIMNAME                 Run a simulation named SIMNAME
    -d, --dimensions=SIZE             Generate a default simulation with SIZE specified e.g. 10x10x10
    -m, --magnetization=DIRECTION     Initial magnetization along DIRECTION specified like 0,0
    -h, --help                        Shows this message
    """)


if __name__ == "__main__":
    read_config_file()

    mode = None
    simname = None
    nx, ny, nz = (None, None, None)
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
        run_simulation(WORKING_DIRECTORY + simname + "/")
    elif mode == "init":
        if theta_0 is None:
            init_simulation(WORKING_DIRECTORY + simname + "/", int(nx), int(ny), int(nz), params=DEFAULT_PARAMS)
            print(f"Default simulation {simname} generated withe default params. \n"
                  f"Lattice has dimensions {nx}x{ny}x{nz} \n"
                  f"Random initial magnetization")
        else:
            init_simulation(WORKING_DIRECTORY + simname + "/", int(nx), int(ny), int(nz), theta_0=int(theta_0),
                            phi_0=int(phi_0), params=DEFAULT_PARAMS)
            print(f"Default simulation {simname} generated withe default params. \n"
                  f"Lattice has dimensions {nx}x{ny}x{nz} \n"
                  f"Initial magnetization ({theta_0},{phi_0})")
    else:
        usage()
        sys.exit(2)

    print("Finished")
