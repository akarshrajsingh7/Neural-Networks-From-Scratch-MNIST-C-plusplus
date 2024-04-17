#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <image_dataset_input> <image_tensor_output> <image_index>"
    exit 1
fi

# Assign input arguments to variables
IMAGE_DATASET_INPUT="$1"
IMAGE_TENSOR_OUTPUT="$2"
IMAGE_INDEX="$3"

# Check if input image dataset exists
if [ ! -f "$IMAGE_DATASET_INPUT" ]; then
    echo "Error: Image dataset not found: $IMAGE_DATASET_INPUT"
    exit 1
fi

## Compile the source files into object files
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/mnist_data_reader.cpp -o ./src/mnist_data_reader.o
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 -c ./src/test_image_reader.cpp -o ./src/test_image_reader.o
#
## Link the object files to create the executable
#g++ -std=c++11 -I /opt/homebrew/include/eigen3 ./src/mnist_data_reader.o ./src/test_image_reader.o -o ./src/read_dataset_images
#
## Run the executable with the provided arguments
#./src/read_dataset_images $IMAGE_DATASET_INPUT $IMAGE_TENSOR_OUTPUT $IMAGE_INDEX

./build/read_dataset_images $IMAGE_DATASET_INPUT $IMAGE_TENSOR_OUTPUT $IMAGE_INDEX