#!/bin/bash

#echo "This script should build your project now..."

# Create a build directory if it doesn't exist
mkdir -p build

# Navigate to the build directory
cd build

# configure project
cmake ..
#cmake .. -DCMAKE_PREFIX_PATH=/builds/advpt-student/ws2023-group-47-overfitters/eigen-3.4.0
#cmake .. -DEigen3_DIR=/builds/advpt-student/ws2023-group-47-overfitters/eigen-3.4.0/cmake/

# build project
make
