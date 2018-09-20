# ThermalSpin

A simple implementation of a Monte Carlo simulation of the classical Heisenberg model on
a 2D or 3D lattice

## Usage

### Run a simulation 

To generate a default lattice with given dimension just run
```lang=bash
$ python3 heisenberg.py --default 10x10x2
```
A directory will be created with default file.
Now just rename the directory and edit the parameters in the json file.
```lang=bash
$ mv ./simulations/default ./simulations/sim_0
$ nano ./simulations/sim_0/params.json
```
Then you are ready to run the simulation with:
```
$ python3 heisenberg.py --run sim_0
```
To analyze the result, just run data_visualization.ipynb in Jupiter.

### Continue a simulation 
Inside the ./simulation directory there will be saved four files:
- params.json       Parameters of the simulation
- state.npy         End state of the simulation
- snapshots.npy     Snapshosts of the sistems at various steps of the simulation 
- snapshots_t.npy   Step number of the snapshots taken

If now run again the simulation with
```
$ python3 heisenberg.py --run sim_0
```

the program will just take the end state as initial state and append the results to the old ones.

# Author
- Andrea Maiani (andrea.maiani@mail.polimi.it)