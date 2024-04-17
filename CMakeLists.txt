cmake_minimum_required(VERSION 3.10)

project(OVERFITTERS)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

set(SOURCE_FILES_RD_DS_IMAGES
        src/mnist_data_reader.cpp
        src/test_image_reader.cpp
)

set(SOURCE_FILES_RD_DS_LABELS
        src/mnist_data_reader.cpp
        src/test_label_reader.cpp
)

set(SOURCE_FILES_MNIST
        src/mnist.cpp
        src/mnist_data_reader.cpp
        src/network.cpp
        src/eval.cpp
        src/Layers/activations.cpp
        src/Layers/fullyconnected.cpp
        src/Optimization/loss.cpp
        src/Optimization/sgd.cpp
)

add_executable(read_dataset_images ${SOURCE_FILES_RD_DS_IMAGES})
target_link_libraries (read_dataset_images Eigen3::Eigen)

add_executable(read_dataset_labels ${SOURCE_FILES_RD_DS_LABELS})
target_link_libraries (read_dataset_labels Eigen3::Eigen)

add_executable(mnist ${SOURCE_FILES_MNIST})
target_link_libraries (mnist Eigen3::Eigen OpenMP::OpenMP_CXX)
