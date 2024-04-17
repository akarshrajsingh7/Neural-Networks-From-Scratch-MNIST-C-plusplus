#include "mnist_data_reader.hpp"
#include "network.hpp"
#include <Eigen/Core>
#include <map>

void writeTensorToFile(const Eigen::VectorXd& tensor, const std::string& filename)
{
    std::ofstream file;
    file.open(filename);

    if (tensor.size() == 10) {
        file << 1 << "\n";
        file << 10 << "\n";
    }
    else {
        file << 2 << "\n";
        file << 28 << "\n";
        file << 28 << "\n";
    }


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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_dataset_input> <image_matrix_output> <image_index>\n";
        return 1;
    }

    std::string imageDatasetInput = argv[1];
    std::string imageMatrixOutput = argv[2];
    int imageIndex = std::stoi(argv[3]);

    MNISTDataReader images_reader(imageDatasetInput);

    Eigen::MatrixXd images = images_reader.readImages();

    Eigen::VectorXd image = images.row(imageIndex);

    writeTensorToFile(image, imageMatrixOutput);

    return 0;
}