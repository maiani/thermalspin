#!/usr/bin/env bash

### Initialize the ferromagnetic LT set
#cp -f ./params_files/3D_config.json ./config.json
#./set_simulation.py -i ferro_set -L 10,12,14,16,18,20 -T 0.5,2.6,0.1 --tilted
#./set_simulation.py -i ferro_set -L 10,12,14,16,18,20 -T 1.4,1.5,0.005 --tilted

## Initialize the ferromagnetic critical set
cp -f ./params_files/3D_config.json ./config.json
./set_simulation.py -i ferro_critic_set -L 10,12,14,16 -T 1.4,1.5,0.005 --tilted

### Initialize the ferromagnetic critical LH set
# cp ./params_files/3D_config.json ./config.json
# ./set_simulation.py -i ferro_critic_LH_set -L 8,10,12,14,16,18,20 -T 1.445 -H 0,1.1,0.1 --tilted

### Initialize the antiferromagnetic LT set
# cp -f ./params_files/3D_config.json ./config.json
# ./set_simulation.py -i antiferro_set -L 10,12,14,16,18,20 -T 0.5,2.6,0.1 -J "-1"
# ./set_simulation.py -i antiferro_set -L 10,12,14,16,18,20 -T 1.4,1.5,0.005 -J "-1"

### Initialize the antiferromagnetic critical LH set
# ./set_simulation.py -i antiferro_critic_set -L 16,18,20,22,24 -T 1.44 -H 0,0.91,0.1 -J "-1"

###  Heisenberg 2D
#cp -f ./params_files/2D_config.json ./config.json
#./set_simulation.py -i heisenberg_2D --2D -L 16,20,24,28,32 -T 0.25,2,0.05 --tilted
