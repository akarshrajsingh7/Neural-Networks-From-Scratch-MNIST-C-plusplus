#pragma once

#include <random>
#include <Eigen/Dense>
#include "../Optimization/sgd.hpp"

class FullyConnected {
private:
    int input_size;
    int output_size;
    Eigen::MatrixXd weights;
    Eigen::MatrixXd _input_matrix;
    SGD optimizer;

public:
    FullyConnected(int input_size, int output_size, int batch_size, float learning_rate);

    FullyConnected(const FullyConnected& other);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input_matrix);

    Eigen::MatrixXd backward(const Eigen::MatrixXd& error_matrix);
};