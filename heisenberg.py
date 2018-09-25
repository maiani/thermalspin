#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
User interface for initializing and run simulations
"""

import getopt
import sys

from heisenberg_simulation import init_simulation, run_simulation

WORKING_DIRECTORY = "./simulations/"


def usage():
    print("""
    Usage: heisenberg.py [OPTIONS] [PARAMETERS]\n
    -i, --init=SIMNAME                Initialize a simulation, need to specify next a dimension and a magnetization
    -r, --run=SIMNAME                 Run a simulation named SIMNAME
    -d, --dimensions=SIZE             Generate a default simulation with SIZE specified e.g. 10x10x10
    -m, --magnetization=DIRECTION     Initial magnetization along DIRECTION specified like 0,0
    -h, --help                        Shows this message
    """)


if __name__ == "__main__":
    # Fallback
    mode = ""
    simname = "default"
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
            init_simulation(WORKING_DIRECTORY + simname + "/", int(nx), int(ny), int(nz))
            print(f"Default simulation {simname} generated withe default params. \n"
                  f"Lattice has dimensions {nx}x{ny}x{nz} \n"
                  f"Random initial magnetization")
        else:
            init_simulation(WORKING_DIRECTORY + simname + "/", int(nx), int(ny), int(nz), theta_0=int(theta_0),
                            phi_0=int(phi_0))
            print(f"Default simulation {simname} generated withe default params. \n"
                  f"Lattice has dimensions {nx}x{ny}x{nz} \n"
                  f"Initial magnetization ({theta_0},{phi_0})")
    else:
        usage()
        sys.exit(2)

    print("Finished")
