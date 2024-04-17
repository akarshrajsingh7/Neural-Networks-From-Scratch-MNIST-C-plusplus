#include "fullyconnected.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <omp.h>

// Constructor for the FullyConnected class
// Initializes the optimizer, input_size, output_size, and _input_matrix
// Also initializes the weights using a normal distribution
FullyConnected::FullyConnected(int input_size, int output_size, int batch_size, float learning_rate) 
    : optimizer(learning_rate), input_size(input_size), output_size(output_size), _input_matrix(Eigen::MatrixXd::Zero(batch_size, input_size+1)) {
    
    std::mt19937 gen(42);  // Seeding

    // Initialize weights with He Initialization
    std::normal_distribution<> dis(0.0, std::sqrt(2.0 / input_size));
    weights = Eigen::MatrixXd(output_size, input_size + 1);
    weights = weights.unaryExpr([&](double x) { return dis(gen); });
}

// Copy constructor for the FullyConnected class
FullyConnected::FullyConnected(const FullyConnected& other)
    : optimizer(other.optimizer), 
      input_size(other.input_size), 
      output_size(other.output_size), 
      _input_matrix(other._input_matrix), 
      weights(other.weights) {}

// Forward pass for the FullyConnected class
Eigen::MatrixXd FullyConnected::forward(const Eigen::MatrixXd& input_matrix) {
    int batch_size = input_matrix.rows();
    Eigen::MatrixXd output_matrix(batch_size, output_size);
    Eigen::VectorXd bias = Eigen::VectorXd::Ones(1);
    Eigen::VectorXd input_with_bias = Eigen::VectorXd::Zero(input_size + 1);
    
    // Add bias to the input
    for (int i = 0; i < batch_size; ++i) {
        input_with_bias.head(input_size) = input_matrix.row(i);
        input_with_bias[input_size] = bias[0];
        _input_matrix.row(i) = input_with_bias;
    }

    // Calculate the output
    output_matrix.noalias() = _input_matrix * weights.transpose();
    
    return output_matrix;
}

// Backward pass for the FullyConnected class
Eigen::MatrixXd FullyConnected::backward(const Eigen::MatrixXd& error_matrix) {
    int batch_size = error_matrix.rows();
    Eigen::MatrixXd gradient_weights = Eigen::MatrixXd::Zero(output_size, input_size + 1);
    Eigen::MatrixXd gradient_inputs = Eigen::MatrixXd::Zero(batch_size, input_size);

    // Calculate the gradient of the weights
    gradient_weights.noalias() = error_matrix.transpose() * _input_matrix;

    // Calculate the gradient of the inputs
    Eigen::MatrixXd slice = weights.leftCols(input_size);
    gradient_inputs.noalias() = error_matrix * slice;

    // Update the weights
    optimizer.update(weights, gradient_weights);
    
    return gradient_inputs;
}