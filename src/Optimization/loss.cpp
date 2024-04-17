#include "loss.hpp"
#include <cmath>
#include <Eigen/Dense>
#include <iostream>

// Default constructor for the CrossEntropyLoss class
CrossEntropyLoss::CrossEntropyLoss() {}

// Copy constructor for the CrossEntropyLoss class
CrossEntropyLoss::CrossEntropyLoss(const CrossEntropyLoss& other) : _prediction_matrix(other._prediction_matrix) {}

// Forward pass for the CrossEntropyLoss class
double CrossEntropyLoss::forward(const Eigen::MatrixXd& prediction_matrix, const Eigen::MatrixXd& label_matrix) {
    // Store the prediction matrix
    _prediction_matrix = prediction_matrix;

    // Create a matrix of small values (epsilon) to avoid log(0)
    Eigen::MatrixXd epsilon = Eigen::MatrixXd::Constant(label_matrix.rows(), label_matrix.cols(), std::numeric_limits<double>::epsilon());

    // Calculate the log of the predictions
    Eigen::MatrixXd log_predictions = (_prediction_matrix.array() + epsilon.array()).log();

    // Calculate the cross entropy loss
    double loss = -(label_matrix.array() * log_predictions.array()).sum();

    return loss;
}

// Backward pass for the CrossEntropyLoss class
Eigen::MatrixXd CrossEntropyLoss::backward(const Eigen::MatrixXd& label_matrix) {
    // Create a matrix of small values (epsilon) to avoid division by 0
    Eigen::MatrixXd epsilon = Eigen::MatrixXd::Constant(label_matrix.rows(), label_matrix.cols(), std::numeric_limits<double>::epsilon());

    // Calculate the error matrix
    Eigen::MatrixXd error_matrix = -label_matrix.array() / (_prediction_matrix.array() + epsilon.array());

    return error_matrix;
}