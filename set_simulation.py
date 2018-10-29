#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run multiple instance of Heisenberg
"""

import getopt
import os
import sys
from multiprocessing.pool import Pool

import numpy as np

from heisenberg_simulation import init_simulation_tilted, init_simulation_aligned, init_simulation_random, \
    run_simulation
from read_config import read_config_file
from skdylib.counter import Counter


def init_set(setname, J, Hz, T, L, theta_0=None, phi_0=None, tilted=False):
    simdir_list = []
    params_list = []
    L_list = []
    set_directory = SIMULATIONS_DIRECTORY + setname + "/"

    for i, j, k in np.ndindex((T.shape[0], L.shape[0], Hz.shape[0])):
        simdir_list.append(set_directory + setname + f"_T{int(T[i]*100)/100}_L{L[j]}_H{int(Hz[k]*100)/100}/")
        params = DEFAULT_PARAMS
        params["param_T"] = [T[i]]
        params["param_Hz"] = [Hz[k]]
        params["param_J"] = J
        params_list.append(params.copy())
        L_list.append(L[j].copy())

    if tilted:
        for i in range(len(simdir_list)):
            init_simulation_tilted(simdir_list[i], nx=L_list[i], ny=L_list[i], nz=L_list[i], params=params_list[i])

    elif theta_0 is None:
        for i in range(len(simdir_list)):
            init_simulation_random(simdir_list[i], nx=L_list[i], ny=L_list[i], nz=L_list[i], params=params_list[i])

    else:
        for i in range(len(simdir_list)):
            init_simulation_aligned(simdir_list[i], nx=L_list[i], ny=L_list[i], nz=L_list[i], params=params_list[i],
                                    theta_0=theta_0, phi_0=phi_0)


simulations_number = 0
completed_simulations_counter = Counter()


def run_simulation_wrapper(simdir):
    run_simulation(simdir, verbose=False)

    completed_simulations_counter.increment()
    completed_simulations_number = completed_simulations_counter.value()

    print(f"Completed simulations {completed_simulations_number}/{simulations_number}")


def run_set(setname):
    filelist = os.listdir(SIMULATIONS_DIRECTORY + setname + "/")
    simdir_list = []

    for filename in filelist:
        if filename.find(setname) >= 0:
            simdir_list.append(SIMULATIONS_DIRECTORY + setname + "/" + filename + "/")

    global simulations_number
    simulations_number = len(simdir_list)

    pool = Pool(processes=PROCESSES_NUMBER)
    pool.map(run_simulation_wrapper, simdir_list)


def usage():
    print("""
    Usage: set_simulation.py [OPTIONS] [PARAMETERS]\n
    -i, --init=SETNAME                Initialize a set of simulations, need to specify next a dimension and a magnetization
    -r, --run=SETNAME                 Run a set of simulations named SETNAME
    -L                                Specify the side dimension of the lattices size like 6,8,10 
    -m, --magnetization=DIRECTION     Initial magnetization along DIRECTION specified like 0,0
    -T, --temperature=TEMP            Specify the range of temperature with TEMP like T_initial,T_final,dT e.g 0.5,3.5,1  
    -H                                Specify the range of external field  like H_initial,H_final, dH
    --tilted                          Tilted initial condition
    -h, --help                        Shows this message
    """)


if __name__ == "__main__":
    DEFAULT_PARAMS, SIMULATIONS_DIRECTORY, PROCESSES_NUMBER = read_config_file()

    mode = None
    setname = None
    L = []
    J = DEFAULT_PARAMS["param_J"]
    T = DEFAULT_PARAMS["param_T"]
    Hz = DEFAULT_PARAMS["param_Hz"]
    theta_0, phi_0 = (None, None)
    Ti, Tf, dt = (None, None, None)
    tilted = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:i:L:m:T:J:H:",
                                   ["help", "initialize=", "run=", "dimensions=", "temperatures=", 'tilted'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
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
            if len(arg.split(",")) == 3:
                Ti, Tf, dT = arg.split(",")
                Ti = float(Ti)
                Tf = float(Tf)
                dT = float(dT)
                T = np.arange(Ti, Tf, dT)
            elif len(arg.split(",")) == 1:
                T = np.array([float(arg)])
        elif opt in ("-H"):
            if len(arg.split(",")) == 3:
                Hi, Hf, dH = arg.split(",")
                Hi = float(Hi)
                Hf = float(Hf)
                dH = float(dH)
                Hz = np.arange(Hi, Hf, dH)
            elif len(arg.split(",")) == 1:
                H = np.array([float(arg)])
        elif opt in "-J":
            J = float(arg)
        elif opt in "--tilted":
            tilted = True

    L = np.array(L)
    if mode == "run":
        print(f"Running simulations in set {setname}")
        run_set(setname)
    elif mode == "init":
        if theta_0 is None:
            init_set(setname, J, Hz, T, L, tilted=tilted)
        else:
            init_set(setname, J, Hz, T, L, theta_0, phi_0, tilted=tilted)
    else:
        usage()
        sys.exit(2)

    print("Finished")
