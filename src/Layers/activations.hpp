#pragma once

#include <Eigen/Dense>

class ReLU {
public:
    ReLU();
    ReLU(const ReLU& other);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& inputs);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& dvalues);

private:
    Eigen::MatrixXd inputs;
};

class Softmax {
public:
    Softmax();
    Softmax(const Softmax& other);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& inputs);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& dvalues);

private:
    Eigen::MatrixXd outputs;
};