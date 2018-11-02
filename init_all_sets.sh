#!/usr/bin/env bash

# Initialize the ferromagnetic LT set
# ./set_simulation.py -i ferro_set -L 10,12,14,16,18,20 -T 0.5,2.6,0.1 --tilted
# ./set_simulation.py -i ferro_set -L 10,12,14,16,18,20 -T 1.4,1.5,0.005 --tilted

# Initialize the ferromagnetic critical LH set
# ./set_simulation.py -i ferro_critic_set -L 10,12,14,16,18,20 -T 1.445 --tilted
# ./set_simulation.py -i ferro_critic_LH_set -L 10,12,14,16,18,20 -T 1.445 -H 0,1.1,0.1 --tilted

# Initialize the ferromagnetic critical LH set
# ./set_simulation.py -i ferro_nc_set -L 10,12,14,16,18,20 -T 1.440,1.451,0.005  --tilted


#Initialize the antiferromagnetic LT set
# ./set_simulation.py -i antiferro_set -L 12,16,18,20,22,24 -T 0.5,2.6,0.1 -J "-1"
# ./set_simulation.py -i antiferro_set -L 12,16,18,20,22,24 -T 1.4,1.51,0.01 -J "-1"

# Initialize the antiferromagnetic critical LH set
# ./set_simulation.py -i antiferro_critic_set -L 16,18,20,22,24 -T 1.44 -H 0,0.91,0.1 -J "-1"


#./set_simulation.py -i heisenberg_2D --2D -L 24 -T 0.3,3,0.1 --tilted
