#pragma once
#include <vector>
#include <stdexcept>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<double> data;

    Tensor(int rows, int cols, const std::vector<double>& input_data) {
        if (input_data.size() != rows * cols) {
            throw std::invalid_argument("Shape and data mismatch");
        }
        shape = {rows, cols};
        data = input_data;
    }

    int get_index(int row, int col) const {
        return (row * shape[1]) + col;
    }

    Tensor matmul(const Tensor& B) const {
        int A_rows = shape[0];
        int A_cols = shape[1];
        int B_rows = B.shape[0];
        int B_cols = B.shape[1];

        if (A_cols != B_rows) {
            throw std::invalid_argument("Matrix dimensions not aligned");
        }

        std::vector<double> C_data(A_rows * B_cols, 0.0);
        Tensor C(A_rows, B_cols, C_data);

        for (int i = 0; i < A_rows; ++i) {
            for (int j = 0; j < B_cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < A_cols; ++k) {
                    sum += data[get_index(i, k)] * B.data[B.get_index(k, j)];
                }
                C.data[C.get_index(i, j)] = sum;
            }
        }
        return C;
    }
};