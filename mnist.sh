#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

# Assign input arguments to variables
CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Label dataset not found: $LABEL_DATASET_INPUT"
    exit 1
fi

## Compile the source files into object files
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/mnist_data_reader.cpp -o ./src/mnist_data_reader.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/Layers/fullyconnected.cpp -o ./src/Layers/fullyconnected.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/Layers/activations.cpp -o ./src/Layers/activations.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/Optimization/loss.cpp -o ./src/Optimization/loss.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/Optimization/sgd.cpp -o ./src/Optimization/sgd.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/network.cpp -o ./src/network.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/eval.cpp -o ./src/eval.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/mnist.cpp -o ./src/mnist.o
#
## Compile the source files into object files
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 ./src/mnist_data_reader.o ./src/Layers/fullyconnected.o ./src/Layers/activations.o ./src/Optimization/loss.o ./src/Optimization/sgd.o ./src/network.o ./src/eval.o ./src/mnist.o -o ./src/mnist
#
## Run the executable with the provided arguments
#./src/mnist $CONFIG_FILE

./build//mnist $CONFIG_FILE