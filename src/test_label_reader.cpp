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
        std::cerr << "Usage: " << argv[0] << " <label_dataset_input> <label_matrix_output> <label_index>\n";
        return 1;
    }

    std::string labelDatasetInput = argv[1];
    std::string labelMatrixOutput = argv[2];
    int labelIndex = std::stoi(argv[3]);

    MNISTDataReader labels_reader(labelDatasetInput);

    Eigen::MatrixXd labels = labels_reader.readLabels();

    Eigen::VectorXd label = labels.row(labelIndex);

    writeTensorToFile(label, labelMatrixOutput);

    return 0;
}