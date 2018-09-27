#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run multiple instance of Heisenberg
"""

import getopt
import json
import os
import sys
from multiprocessing.pool import Pool

import numpy as np

from heisenberg_simulation import init_simulation, run_simulation

CONFIG_FILE_NAME = "config.json"

PROCESSES_NUMBER = None
WORKING_DIRECTORY = None
DEFAULT_PARAMS = None


def init_set(setname, T, L, theta_0=None, phi_0=None, ):
    simdir_list = []
    params_list = []
    L_list = []

    for i, j in np.ndindex((T.shape[0], L.shape[0])):
        simdir_list.append(WORKING_DIRECTORY + setname + "/" + setname + "_T" + str(T[i]) + "_L" + str(L[j]) + "/")
        params = DEFAULT_PARAMS
        params["T"] = T[i]
        params_list.append(params.copy())
        L_list.append(L[j].copy())

    for i in range(len(simdir_list)):
        init_simulation(simdir_list[i], L_list[i], L_list[i], L_list[i], params=params_list[i], theta_0=theta_0,
                        phi_0=phi_0)


def run_set(setname):
    filelist = os.listdir(WORKING_DIRECTORY + setname + "/")
    simdir_list = []

    for filename in filelist:
        if filename.find(setname) >= 0:
            simdir_list.append(WORKING_DIRECTORY + setname + "/" + filename + "/")

    pool = Pool(processes=PROCESSES_NUMBER)
    pool.map(run_simulation, simdir_list)


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
    Usage: heisenberg.py [OPTIONS] [PARAMETERS]\n
    -i, --init=SETNAME                Initialize a set of simulations, need to specify next a dimension and a magnetization
    -r, --run=SETNAME                 Run a set of simulations named SETNAME
    -L                                Specify the side dimension of the lattices size like 6,8,10 
    -m, --magnetization=DIRECTION     Initial magnetization along DIRECTION specified like 0,0
    -T, --temperature=TEMP            Specify the range of temperature with TEMP like T_initial,T_final,dT e.g 0.5,3.5,1  
    -h, --help                        Shows this message
    """)


if __name__ == "__main__":
    mode = None
    setname = None
    L = []
    T = []
    theta_0, phi_0 = (None, None)
    Ti, Tf, dt = (None, None, None)

    read_config_file()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:i:L:m:T:",
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
        elif opt in ("-L"):
            for dim in arg.split(","):
                L.append(int(dim))
        elif opt in ("-m", "--magnetization"):
            theta_0, phi_0 = arg.split(",")
        elif opt in ("-T", "--temperature"):
            Ti, Tf, dT = arg.split(",")
            Ti = float(Ti)
            Tf = float(Tf)
            dT = float(dT)
            T = np.arange(Ti, Tf + 00.1, dT)

    L = np.array(L)
    if mode == "run":
        print(f"Running simulation in set {setname}")
        run_set(setname)
    elif mode == "init":
        if theta_0 is None:
            init_set(setname, T, L)
        else:
            init_set(setname, T, L, theta_0, phi_0)
    else:
        usage()
        sys.exit(2)

    print("Finished")
