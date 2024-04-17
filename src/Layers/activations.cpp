#include "activations.hpp"
#include <iostream>
#include <Eigen/Dense>

// Default constructor for the ReLU class
ReLU::ReLU() {}

// Copy constructor for the ReLU class
ReLU::ReLU(const ReLU& other) : inputs(other.inputs) {}

// Forward pass for the ReLU class
Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& inputs) {
    // Store the inputs
    this->inputs = inputs;
    // Apply the ReLU function to the inputs
    return inputs.unaryExpr([](double elem) { return std::max(0.0, elem); });
}

// Backward pass for the ReLU class
Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& dvalues) {
    // Calculate the gradients
    Eigen::MatrixXd gradients = inputs.unaryExpr([](double v) { return v > 0.0 ? 1.0 : 0.0; });
    // Return the product of the gradients and the upstream gradients
    return gradients.cwiseProduct(dvalues);
}


// Default constructor for the Softmax class
Softmax::Softmax() {}

// Copy constructor for the Softmax class
Softmax::Softmax(const Softmax& other) : outputs(other.outputs) {}

// Forward pass for the Softmax class
Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd& inputs) {
    // Calculate the maximum of each row
    Eigen::VectorXd maxCoeff = inputs.rowwise().maxCoeff();
    // Subtract the maximum from each element (for numerical stability), exponentiate, and sum each row
    Eigen::MatrixXd exp = (inputs.colwise() - maxCoeff).array().exp();
    Eigen::VectorXd sum = exp.rowwise().sum();
    // Divide each element by the sum of its row to get the softmax outputs
    this->outputs = exp.array().colwise() / sum.array();
    return this->outputs;
}

// Backward pass for the Softmax class
Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd& dvalues) {
    // Calculate the product of the upstream gradients and the softmax outputs
    Eigen::MatrixXd product = dvalues.cwiseProduct(this->outputs);
    // Sum each row
    Eigen::VectorXd result = product.rowwise().sum();
    // Subtract the sum from the upstream gradients, multiply by the softmax outputs, and return the result
    Eigen::MatrixXd dinputs = this->outputs.array() * (dvalues.array() - result.replicate(1, dvalues.cols()).array());
    return dinputs;
}