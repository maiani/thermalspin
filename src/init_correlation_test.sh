#!/usr/bin/env bash

echo "Initializing the 3D correlation test"
dynamic_simulation.py -i test14 -d 14x14x14
dynamic_simulation.py -i test16 -d 16x16x16
dynamic_simulation.py -i test18 -d 18x18x18
dynamic_simulation.py -i test20 -d 20x20x20
dynamic_simulation.py -i test22 -d 22x22x22

cp ./params_files/correlation_test.json ./simulations/test14/params.json
cp ./params_files/correlation_test.json ./simulations/test16/params.json
cp ./params_files/correlation_test.json ./simulations/test18/params.json
cp ./params_files/correlation_test.json ./simulations/test20/params.json
cp ./params_files/correlation_test.json ./simulations/test22/params.json

mkdir ./simulations/correlation_test/
mv ./simulations/test14 ./simulations/correlation_test/test14/
mv ./simulations/test16 ./simulations/correlation_test/test16/
mv ./simulations/test18 ./simulations/correlation_test/test18/
mv ./simulations/test20 ./simulations/correlation_test/test20/
mv ./simulations/test22 ./simulations/correlation_test/test22/

echo "Finished"


echo "Initializing the 2D correlation test"
dynamic_simulation.py -i test14_2D -d 14x14x1
dynamic_simulation.py -i test16_2D -d 16x16x1
dynamic_simulation.py -i test18_2D -d 18x18x1
dynamic_simulation.py -i test20_2D -d 20x20x1
dynamic_simulation.py -i test22_2D -d 22x22x1

cp ./params_files/correlation_test.json ./simulations/test14_2D/params.json
cp ./params_files/correlation_test.json ./simulations/test16_2D/params.json
cp ./params_files/correlation_test.json ./simulations/test18_2D/params.json
cp ./params_files/correlation_test.json ./simulations/test20_2D/params.json
cp ./params_files/correlation_test.json ./simulations/test22_2D/params.json

mkdir ./simulations/correlation_test_2D/
mv ./simulations/test14_2D ./simulations/correlation_test_2D/test14_2D/
mv ./simulations/test16_2D ./simulations/correlation_test_2D/test16_2D/
mv ./simulations/test18_2D ./simulations/correlation_test_2D/test18_2D/
mv ./simulations/test20_2D ./simulations/correlation_test_2D/test20_2D/
mv ./simulations/test22_2D ./simulations/correlation_test_2D/test22_2D/

echo "Finished"
