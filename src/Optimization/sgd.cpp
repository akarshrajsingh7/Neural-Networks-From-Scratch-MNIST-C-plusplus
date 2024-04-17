#include "sgd.hpp"
#include <iostream>
#include <Eigen/Dense>

// Constructor for the SGD class
// Initializes learning_rate, momentum_rate, and V
SGD::SGD(double learning_rate, double momentum_rate)
    : learning_rate(learning_rate), momentum_rate(momentum_rate), V(nullptr) {}

// Function to update the weights using SGD with momentum
void SGD::update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& dweights) {
    // If V is not initialized (i.e., this is the first update),
    // initialize it as the negative gradient scaled by the learning rate
    if (V == nullptr) {
        V = new Eigen::MatrixXd(-learning_rate * dweights);
    } else {
        // Otherwise, update V based on its previous value and the current gradient
        *V = momentum_rate * (*V) - (learning_rate * dweights);
    }
    // Update the weights by adding V
    weights = weights + (*V);
}