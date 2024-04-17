#pragma once

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

class MNISTDataReader {
private:
    std::ifstream file;

public:
    MNISTDataReader(const std::string& filename);

    ~MNISTDataReader();

    Eigen::MatrixXd readImages();

    Eigen::MatrixXd readLabels();
};