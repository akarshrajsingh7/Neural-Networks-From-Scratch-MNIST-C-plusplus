#include "network.hpp"
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;

// Network constructor
Network::Network(const Eigen::MatrixXd& images, const Eigen::MatrixXd& labels, int input_size, int output_size, int hidden_size, int batch_size, float learning_rate, int num_threads)
    : _images(images), _labels(labels), _input_size(input_size), _hidden_size(hidden_size), _output_size(output_size), _batch_size(batch_size),
      _fc1s(num_threads, FullyConnected(input_size, hidden_size, batch_size, learning_rate)), 
      _relus(num_threads, ReLU()), 
      _fc2s(num_threads, FullyConnected(hidden_size, output_size, batch_size, learning_rate)), 
      _softmaxs(num_threads, Softmax()), 
      _cross_entropy_losses(num_threads, CrossEntropyLoss()) {}

// Forward pass of the network
Eigen::MatrixXd Network::forward(const Eigen::MatrixXd& input_matrix) {
    FullyConnected& fc1 = _fc1s[omp_get_thread_num()];
    ReLU& relu = _relus[omp_get_thread_num()];
    FullyConnected& fc2 = _fc2s[omp_get_thread_num()];
    Softmax& softmax = _softmaxs[omp_get_thread_num()];

    //FC1 -> ReLU -> FC2 -> Softmax
    Eigen::MatrixXd output_fc1 = fc1.forward(input_matrix);
    Eigen::MatrixXd output_relu = relu.forward(output_fc1);
    Eigen::MatrixXd output_fc2 = fc2.forward(output_relu);
    Eigen::MatrixXd output_softmax = softmax.forward(output_fc2);
    return output_softmax;
}

// Backward pass of the network
void Network::backward(const Eigen::MatrixXd& label_matrix) {
    CrossEntropyLoss& cross_entropy_loss = _cross_entropy_losses[omp_get_thread_num()];
    Softmax& softmax = _softmaxs[omp_get_thread_num()];
    FullyConnected& fc2 = _fc2s[omp_get_thread_num()];
    ReLU& relu = _relus[omp_get_thread_num()];
    FullyConnected& fc1 = _fc1s[omp_get_thread_num()];

    //Softmax -> FC2 -> ReLU -> FC1 (Backward Pass)
    Eigen::MatrixXd error_matrix = cross_entropy_loss.backward(label_matrix);
    error_matrix = softmax.backward(error_matrix);
    error_matrix = fc2.backward(error_matrix);
    error_matrix = relu.backward(error_matrix);
    error_matrix = fc1.backward(error_matrix);
}

// Training function
void Network::train(const int epochs) {
    omp_set_num_threads(6);
    Eigen::setNbThreads(6);
    std::cout << "Number of threads using Eigen (Network): " << Eigen::nbThreads() << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();

        double loss = 0.0;
        double accuracy = 0.0;

        // Parallelize the training process
        #pragma omp parallel for reduction(+:loss,accuracy)
        for (int i = 0; i < _images.rows() / _batch_size; ++i) {
            Eigen::MatrixXd images_batch = _images.block(i * _batch_size, 0, _batch_size, _input_size);
            Eigen::MatrixXd labels_batch = _labels.block(i * _batch_size, 0, _batch_size, _output_size);

            Eigen::MatrixXd prediction = forward(images_batch);
            
            CrossEntropyLoss& cross_entropy_loss = _cross_entropy_losses[omp_get_thread_num()];
            // Accumulating loss
            loss += cross_entropy_loss.forward(prediction, labels_batch);
            // Calculating accuracy
            accuracy += eval.calculate_accuracy(prediction, labels_batch);

            // Backward pass
            backward(labels_batch);
        }
        // Calculating average loss and accuracy over the epoch
        double avg_loss = loss / (_images.rows() / _batch_size);
        double avg_accuracy = accuracy / (_images.rows() / _batch_size);
        this->losses.push_back(avg_loss);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "Epoch " << epoch << " : Loss: " << avg_loss << std::endl;
        std::cout << "Epoch " << epoch << " : Avg Accuracy: " << avg_accuracy << std::endl;
        std::cout << "Time taken in Epoch " << epoch << " : " << elapsed.count() << " s" << std::endl;
    }
}

// Testing function
Eigen::MatrixXd Network::test(const Eigen::MatrixXd& input_matrix, const Eigen::MatrixXd& labels_matrix){
    std::cout << "Testing" << std::endl;
    Eigen::MatrixXd images_batch(_batch_size, _input_size);
    Eigen::MatrixXd labels_batch(_batch_size, _output_size);
    Eigen::MatrixXd predictions(labels_matrix.rows(), _output_size);
    double accuracy = 0.0;
    for (int i = 0; i < input_matrix.rows() / _batch_size; ++i) {
        images_batch = input_matrix.block(i * _batch_size, 0, _batch_size, _input_size);
        labels_batch = labels_matrix.block(i * _batch_size, 0, _batch_size, _output_size);

        Eigen::MatrixXd prediction = forward(images_batch);

        predictions.block(i * _batch_size, 0, _batch_size, _output_size) = prediction;
        
        accuracy += eval.calculate_accuracy(prediction, labels_batch);
    }
    double avg_accuracy = accuracy / (input_matrix.rows() / _batch_size);
    
    std::cout << "Test Accuracy: " << avg_accuracy << std::endl;

    return predictions;
}