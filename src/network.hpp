#pragma once

#include "Layers/fullyconnected.hpp"
#include "Layers/activations.hpp"
#include "Optimization/loss.hpp"
#include "eval.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

class Network {
public:
    Network(const Eigen::MatrixXd& images, const Eigen::MatrixXd& labels, int input_size, int output_size, int hidden_size, int batch_size, float learning_rate, int num_threads);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input_matrix);
    void backward(const Eigen::MatrixXd& label_matrix);
    void train(const int epochs);
    Eigen::MatrixXd test(const Eigen::MatrixXd& input_matrix, const Eigen::MatrixXd& labels_matrix);

private:
    int _input_size, _hidden_size, _output_size, _batch_size;
    std::vector<double> losses;
    Eigen::MatrixXd _images;
    Eigen::MatrixXd _labels;

    std::vector<FullyConnected> _fc1s;
    std::vector<ReLU> _relus;
    std::vector<FullyConnected> _fc2s;
    std::vector<Softmax> _softmaxs;
    std::vector<CrossEntropyLoss> _cross_entropy_losses;
    Eval eval;
};