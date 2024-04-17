#include "mnist_data_reader.hpp"
#include "network.hpp"
#include <Eigen/Core>
#include "eval.hpp"
#include <map>
#include <omp.h>

// Function to write a tensor to a file
void writeTensorToFile(const Eigen::VectorXd& tensor, const std::string& filename)
{
    std::ofstream file;
    file.open(filename);

    // Write the dimensions of the tensor to the file
    if (tensor.size() == 10) {
        file << 1 << "\n";
        file << 10 << "\n";
    }
    else {
        file << 2 << "\n";
        file << 28 << "\n";
        file << 28 << "\n";
    }

    // Write the elements of the tensor to the file
    if (tensor.size() == 1)
    {
        file << tensor({}) << "\n";
    }
    else {
        for (int i = 0; i < tensor.size(); ++i) {
            file << tensor(i) << "\n";
        }
    }

    file.close();
}

// Function to read a configuration file
std::map<std::string, std::string> readConfigFile(const std::string& filename) {
    std::map<std::string, std::string> configMap;

    // Open the config file
    std::ifstream configFile(filename);
    if (!configFile.is_open()) {
        std::cerr << "Failed to open the config file: " << filename << std::endl;
        return configMap; // Return empty map
    }

    // Read the file line by line
    std::string line;
    while (std::getline(configFile, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;

        // Find the position of the '=' character
        std::size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            std::cerr << "Invalid line in config file: " << line << std::endl;
            continue;
        }

        // Extract key and value
        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);

        // Trim leading and trailing whitespaces from key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // Insert into the map
        configMap[key] = value;
    }

    // Close the config file
    configFile.close();

    return configMap;
}

// Function to write the predictions and labels to a logger file
void writeLoggerFile(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& labels, const std::string& filename, int batch_size){
    std::ofstream file;
    file.open(filename);

    // Write the predictions and labels to the file
    for (int i = 0; i < predictions.rows(); ++i){
        if (i % batch_size == 0){
            file << "Current batch: " << i / batch_size << "\n";
        }

        int predicted_class = 0, label_class = 0;
        double max_prob = predictions(i, 0);
        for (int j = 1; j < predictions.cols(); ++j) {
            if (predictions(i, j) > max_prob) {
                max_prob = predictions(i, j);
                predicted_class = j;
            }
            if (labels(i, j) == 1.0) {
                label_class = j;
            }
        }
        file << " - image " << i << ": Prediction=" << predicted_class << ". Label=" << label_class << "\n";
    }

    file.close();
}

// Main function
int main(int argc, char* argv[]) {
    // Set the number of threads for OpenMP and Eigen
    omp_set_num_threads(6);
    Eigen::setNbThreads(6);
    std::cout << "Number of threads using Eigen (MNIST): " << Eigen::nbThreads() << std::endl;

    // Check if a configuration file was provided
    if(argc < 2) {
        std::cout << "Usage: ./program <config_file_path>" << std::endl;
        return 1;
    }

    // Read the configuration file
    std::string config_file = argv[1];
    std::map<std::string, std::string> configMap = readConfigFile(config_file);

    // Extract the configuration parameters
    int batch_size = std::stoi(configMap["batch_size"]);
    int hidden_size = std::stoi(configMap["hidden_size"]);
    float learning_rate = std::stof(configMap["learning_rate"]);
    int num_epochs = std::stoi(configMap["num_epochs"]);

    // Extract the file paths
    std::string rel_path_train_images = configMap["rel_path_train_images"];
    std::string rel_path_train_labels = configMap["rel_path_train_labels"];
    std::string rel_path_test_images = configMap["rel_path_test_images"];
    std::string rel_path_test_labels = configMap["rel_path_test_labels"];
    std::string rel_path_log_file = configMap["rel_path_log_file"];

    // Read the training and testing data
    MNISTDataReader train_images_reader(rel_path_train_images);
    MNISTDataReader train_labels_reader(rel_path_train_labels);
    MNISTDataReader test_images_reader(rel_path_test_images);
    MNISTDataReader test_labels_reader(rel_path_test_labels);

    Eigen::MatrixXd train_images = train_images_reader.readImages();
    Eigen::MatrixXd train_labels = train_labels_reader.readLabels();
    Eigen::MatrixXd test_images = test_images_reader.readImages();
    Eigen::MatrixXd test_labels = test_labels_reader.readLabels();

    // Initialize the network and train it
    int s_train = train_images.rows();
    int s_test = test_images.rows(); 
    int r = train_images.cols(); 
    int output_size = train_labels.cols();
    
    Network network(train_images, train_labels, r, output_size, hidden_size, batch_size, learning_rate, Eigen::nbThreads());
    network.train(num_epochs);

    // Test the network and write the predictions to the logger file
    Eigen::MatrixXd predictions = network.test(test_images, test_labels);
    writeLoggerFile(predictions, test_labels, rel_path_log_file, batch_size);

    return 0;
}