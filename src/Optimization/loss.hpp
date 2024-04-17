#pragma once

#include <Eigen/Dense>

class CrossEntropyLoss {
public:
    CrossEntropyLoss();
    CrossEntropyLoss(const CrossEntropyLoss& other);
    double forward(const Eigen::MatrixXd& prediction_matrix, const Eigen::MatrixXd& label_matrix);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& label_matrix);

private:
    Eigen::MatrixXd _prediction_matrix;
};