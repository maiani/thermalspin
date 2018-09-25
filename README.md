# ThermalSpin

A simple implementation of a Monte Carlo simulation of the classical Heisenberg model on
a 2D or 3D lattice.

The program use Numpy for the linear algebra and Numba jit compiler to accelerate the simulation.

The output can be viewed through a Jupyter notebook.

## Usage

### Run a single simulation 

To generate a default lattice with given dimension just run
```lang=bash
$ python3 heisenberg.py -i sim_16 -d 16x16x16 -m 0,0
```
A simulation directory will be created inside `./simulations` with a default parameter file
and a initialized lattice of the given dimension with spin oriented toward z-axis.
To initialize a lattice of spin random oriented:
```lang=bash
$ python3 heisenberg.py -i sim_16 -d 16x16x16
```

After have edit the parameters in the json file, you are ready to run the simulation with:
```
$ python3 heisenberg.py -r sim_16
```
To analyze the result, just run `data_visualization.ipynb` in Jupiter.

Inside the ./simulation directory there will be saved four files:
- `params.json`            Parameters of the simulation
- `state.npy`              End state of the simulation
- `snapshots.npy`          Snapshosts of the sistems at various steps of the simulation 
- `snapshots_params.npy`   Parameters when the snapshosts were taken
- `results.npy`            Energy and magnetization of each snapshot

If now run again the simulation with
```
$ python3 heisenberg.py --r sim_16
```

the program will just take the end state as initial state and append the results to the old ones.

### Run a set of simulations

To initialize a set of system at different temperature, for example from 0.5 to 3 with a deltaT of 1.5:

```lang=bash
$ python3 heisenberg_set.py -i sim_set -d 16x16x16 -t 0.5,3,0.5
```

And then to run the set:

```
$ python3 heisenberg.py -r sim_set
```
To analyze the result, just run `set_data_visualization.ipynb` in Jupyter.



# Author
- Andrea Maiani (andrea.maiani@mail.polimi.it)