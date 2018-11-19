#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read the configuration file
"""

import json
import os

CONFIG_FILE_NAME = "config.json"


def read_config_file():

    if os.path.isfile(CONFIG_FILE_NAME):
        config_file = open(CONFIG_FILE_NAME, "r")
        config = json.load(config_file)
    else:
        raise Exception("Missing config.json file")

    processes_number = int(config["process_number"])
    simulations_directory = config["simulations_directory"]
    default_param_J = config["default_param_J"]
    default_param_D = config["default_param_D"]
    default_param_Hz = config["default_param_Hz"]
    default_param_T = config["default_param_T"]
    default_steps_number = int(config["default_steps_number"])
    default_delta_snapshots = int(config["default_delta_snapshots"])
    default_save_snapshots = bool(config["default_save_snapshots"])

    default_params = dict(param_J=default_param_J, param_D=default_param_D, param_Hz=default_param_Hz,
                          param_T=default_param_T, steps_number=default_steps_number,
                          delta_snapshots=default_delta_snapshots, save_snapshots=default_save_snapshots)
    return default_params, simulations_directory, processes_number
