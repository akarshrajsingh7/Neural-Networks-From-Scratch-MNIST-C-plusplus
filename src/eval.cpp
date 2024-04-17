#include "eval.hpp"

// Function to calculate the accuracy of the model's predictions
double Eval::calculate_accuracy(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& labels) {
    // Get the number of samples and classes from the predictions matrix
    int num_samples = predictions.rows();
    int num_classes = predictions.cols();
    // Initialize a counter for the number of correct predictions
    int num_correct = 0;

    // Loop over each sample
    for (int i = 0; i < num_samples; ++i) {
        // Initialize the predicted class as the first class
        int predicted_class = 0;
        // Initialize the maximum probability as the probability of the first class
        double max_prob = predictions(i, 0);

        // Loop over each class
        for (int j = 1; j < num_classes; ++j) {
            // If the probability of this class is greater than the current maximum,
            // update the maximum probability and the predicted class
            if (predictions(i, j) > max_prob) {
                max_prob = predictions(i, j);
                predicted_class = j;
            }
        }

        // If the predicted class is the correct class, increment the counter
        if (labels(i, predicted_class) == 1.0) {
            num_correct++;
        }
    }

    // Calculate the accuracy as the number of correct predictions divided by the total number of samples
    double accuracy = static_cast<double>(num_correct) / num_samples;
    return accuracy;
}