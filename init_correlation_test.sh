#!/usr/bin/env bash

echo "Initializing the simulations"
./dynamic_simulation.py -i test18 -d 18x18x18
./dynamic_simulation.py -i test20 -d 20x20x20
./dynamic_simulation.py -i test22 -d 22x22x22
./dynamic_simulation.py -i test24 -d 24x24x24

echo "Copying the params files"
cp ./params_files/correlation_test.json ./simulations/test18/params.json
cp ./params_files/correlation_test.json ./simulations/test20/params.json
cp ./params_files/correlation_test.json ./simulations/test22/params.json
cp ./params_files/correlation_test.json ./simulations/test24/params.json

echo "Creating the correlation_test systems set"
mkdir ./simulations/correlation_test/
mv ./simulations/test18 ./simulations/correlation_test/test18/
mv ./simulations/test20 ./simulations/correlation_test/test20/
mv ./simulations/test22 ./simulations/correlation_test/test22/
mv ./simulations/test24 ./simulations/correlation_test/test24/

echo "Finished"
