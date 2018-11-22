# ThermalSpin

A simple implementation of a Monte Carlo simulation of the classical Heisenberg model on
a 2D or 3D lattice. It is possible to add also antisymmetric DMI interactions to simulate other
type of systems.


The program use Numpy for the linear algebra and Numba jit compiler to accelerate the simulation.

The output can be viewed through a Jupyter notebook.

## Usage

### Configure the program

The configuration of the program is found in the file `config.json`. The important parameters
to set are the number of processes (which should be equal to the number of the core of the computer),
the number of step to be performed and the number of steps between two snapshots.


### Run a set of simulations

To run a set of simulations of a cubic system with cyclic boundary conditions you have to 
select a name for the set and a range of temperature and dimensions.

To initialize the set of system you have to run a command like:

```lang=bash
$ ./set_simulation.py -i sim_816 -L 8,12,16 -T 0.5,3,0.25
```

This will create a set named `sim_816` with dimensions 8, 12 and 16 and temperature ranging
from 0.5 to 3 with steps of 0.25

To run the set:

```
$ ./set_simulation.py -r sim_816
```

Inside the `./simulation/sim_816` a directory for each ensemble will be created.
Inside each one there will be saved four files:
- `params.json`            Parameters of the simulation
- `state.npy`              End state of the simulation
- `snapshots_params.npy`   Parameters when the snapshosts were taken
- `results.npy`            Energy and magnetization of each snapshot

To analyze the result, just run `LT_set_analysis.ipynb` in Jupiter.

It possible to continue the simulation with
```
$ ./set_simulation.py -r sim_816
```

the program will just take the end state as initial state and append the results to the old ones.

### Run a single simulation with dynamic parameters

To generate a default lattice with given dimension just run

```lang=bash
$ ./ dynamic_simulation.py -i sim_16 -d 16x16x16 -m 0,0,1
```
A simulation directory will be created inside `./simulations` with a default parameter file
and a initialized lattice of the given dimension with spin oriented toward z-axis.

To initialize a lattice of spin random oriented:
```lang=bash
$ ./ dynamic_simulation.py -i sim_16 -d 16x16x16
```

To change dynamically the parameters, fill the `param_J`, `param_D`, `param_Hz`, `param_T` variables 
with an array of values. The simulation will proceed running a number of step equals to
`steps_number` with each triplet of parameters in the arrays.

After have edit the parameters in the json file, you are ready to run the simulation with:
```
$ ./ dynamic_simulation.py -r sim_16
```

To visualize the results you can use `dynamic_analysis.ipynb` Jupyter notebook.

Some precompiled `params.json` can be found in the `params_files` folder. Just copy one 
in the initialized simulation folder, change the name in `params.json` and run the simulation. 

### Predefined systems

In the `src` folder two bash files, `init_all_sets.sh` and `init_correlation_test.sh` are available 
to generate some useful standard systems. Just uncomment the rows relative to the system you want to 
initialize and run the files.

## Spherical coordinate version

An old version of the program which use spherical coordinates can be found in `old_sph_version` directory.

# Author
- Andrea Maiani (andrea.maiani@mail.polimi.it)