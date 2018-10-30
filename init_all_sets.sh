#!/usr/bin/env bash

# Initialize the ferromagnetic LT set
./set_simulation.py -i ferro_set -L 16,18,20,22,24 -T 0.4,3.1,0.1 --tilted

# Initialize the ferromagnetic near critical LT set
./set_simulation.py -i ferro_nc_set -L 16,18,20,22,24 -T 1.4,1.51,0.01 --tilted

# Initialize the ferromagnetic critical LH set
./set_simulation.py -i ferro_critic_set -L 16,18,20,22,24 -T 1.44 -H 0,0.91,0.1 --tilted

#Initialize the antiferromagnetic LT set
./set_simulation.py -i antiferro_set -L 16,18,20,22,24 -T 0.4,3.1,0.1 -J "-1"

# Initialize the antiferromagnetic critical LH set
./set_simulation.py -i antiferro_critic_set -L 16,18,20,22,24 -T 1.44 -H 0,0.91,0.1 -J "-1"

