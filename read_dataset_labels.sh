#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <label_dataset_input> <label_tensor_output> <label_index>"
    exit 1
fi

# Assign input arguments to variables
LABEL_DATASET_INPUT="$1"
LABEL_TENSOR_OUTPUT="$2"
LABEL_INDEX="$3"

# Check if input label dataset exists
if [ ! -f "$LABEL_DATASET_INPUT" ]; then
    echo "Error: Label dataset not found: $LABEL_DATASET_INPUT"
    exit 1
fi

## Compile the source files into object files
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/mnist_data_reader.cpp -o ./src/mnist_data_reader.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/test_label_reader.cpp -o ./src/test_label_reader.o
#
## Compile the source files into object files
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 ./src/mnist_data_reader.o ./src/test_label_reader.o -o ./src/read_dataset_labels
#
## Run the executable with the provided arguments
#./src/read_dataset_labels $LABEL_DATASET_INPUT $LABEL_TENSOR_OUTPUT $LABEL_INDEX

./build/read_dataset_labels $LABEL_DATASET_INPUT $LABEL_TENSOR_OUTPUT $LABEL_INDEX