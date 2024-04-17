#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

class Eval{
public:
    double calculate_accuracy(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& labels);
};