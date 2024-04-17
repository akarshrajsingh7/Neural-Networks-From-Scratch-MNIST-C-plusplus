#include "mnist_data_reader.hpp"
#include <Eigen/Dense>

// Constructor for the MNISTDataReader class
// Opens the file specified by the filename parameter
MNISTDataReader::MNISTDataReader(const std::string& filename) {
    file.open(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}

// Destructor for the MNISTDataReader class
// Closes the file
MNISTDataReader::~MNISTDataReader() {
    file.close();
}

// Function to read images from the MNIST dataset
Eigen::MatrixXd MNISTDataReader::readImages() {
    int32_t magic_number = 0, n_images = 0, rows = 0, cols = 0;
    
    // Read the magic number from the file
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(int32_t));
    magic_number = __builtin_bswap32(magic_number);

    // Check if the magic number is correct
    if (magic_number != 2051) {
        std::cerr << "Invalid magic number, not a MNIST image file" << std::endl;
        return {};
    }

    // Read the number of images, rows, and columns from the file
    file.read(reinterpret_cast<char*>(&n_images), sizeof(int32_t));
    n_images = __builtin_bswap32(n_images);

    file.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    rows = __builtin_bswap32(rows);

    file.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    cols = __builtin_bswap32(cols);

    // Initialize a matrix to store the images
    Eigen::MatrixXd images(n_images, rows * cols);
    
    // Read each image from the file
    for (int i = 0; i < n_images; ++i) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                images(i, r * cols + c) = static_cast<double>(pixel) / 255.0;
            }
        }
    }

    return images;
}

// Function to read labels from the MNIST dataset
Eigen::MatrixXd MNISTDataReader::readLabels() {
    int32_t magic_number = 0, n_labels = 0;
    
    // Read the magic number from the file
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(int32_t));
    magic_number = __builtin_bswap32(magic_number);

    // Check if the magic number is correct
    if (magic_number != 2049) {
        std::cerr << "Invalid magic number, not a MNIST label file" << std::endl;
        return {};
    }

    // Read the number of labels from the file
    file.read(reinterpret_cast<char*>(&n_labels), sizeof(int32_t));
    n_labels = __builtin_bswap32(n_labels);

    // Initialize a matrix to store the labels
    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(n_labels, 10);

    // Read each label from the file
    for (int i = 0; i < n_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));

        // Set the corresponding element of the labels matrix to 1
        labels(i, label) = 1.0; 
    }

    return labels;
}