#pragma once

#include <Eigen/Dense>

class SGD {
public:
    SGD(double learning_rate, double momentum_rate=0.9);

    void update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& dweights);
private:
    double learning_rate;
    double momentum_rate;
    Eigen::MatrixXd* V;
};