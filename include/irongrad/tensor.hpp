#pragma once
#include <vector>
#include <stdexcept>
#include <memory>
#include <functional>
#include <unordered_set>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<int> shape;
    std::vector<double> data;
    std::vector<double> grad;

    // Parents in the computation graph
    std::vector<std::shared_ptr<Tensor>> _prev;
    std::function<void()> _backward;

    Tensor(int rows, int cols, const std::vector<double>& input_data) {
        if (input_data.size() != (size_t)(rows * cols)) {
            throw std::invalid_argument("Shape and data mismatch");
        }
        shape = {rows, cols};
        data = input_data;
        grad.resize(data.size(), 0.0);
        _backward = [](){};
    }

    Tensor(int rows, int cols, const std::vector<double>& input_data,
           std::vector<std::shared_ptr<Tensor>> prev) {
        if (input_data.size() != (size_t)(rows * cols)) {
            throw std::invalid_argument("Shape and data mismatch");
        }
        shape = {rows, cols};
        data = input_data;
        grad.resize(data.size(), 0.0);
        _prev = prev;
        _backward = [](){};
    }

    int get_index(int row, int col) const {
        return (row * shape[1]) + col;
    }

    static void backward_matmul_dA(Tensor& A, const Tensor& B, const Tensor& C) {
        int dC_rows = C.shape[0];
        int dC_cols = C.shape[1];

        int B_T_cols = B.shape[0];

        for (int i = 0; i < dC_rows; ++i) {
            for (int j = 0; j < B_T_cols; ++j) {
                double sum = 0.0;

                for (int k = 0; k < dC_cols; ++k) {
                    double dC_val = C.grad[C.get_index(i, k)];
                    // VIRTUAL TRANSPOSE
                    double b_T_val = B.data[B.get_index(j, k)];
                    sum += dC_val * b_T_val;
                }

                A.grad[A.get_index(i, j)] += sum;
            }
        }
    }

    static void backward_matmul_dB(const Tensor& A, Tensor& B, const Tensor& C) {
        int A_T_rows = A.shape[1];

        int dC_rows = C.shape[0];
        int dC_cols = C.shape[1];

        for (int i = 0; i < A_T_rows; ++i) {
            for (int j = 0; j < dC_cols; ++j) {
                double sum = 0.0;

                for (int k = 0; k < dC_rows; ++k) {
                    double a_T_val = A.data[A.get_index(k, i)];
                    double dC_val = C.grad[C.get_index(k, j)];
                    sum += a_T_val * dC_val;
                }

                B.grad[B.get_index(i, j)] += sum;
            }
        }
    }

    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> B) {
        int A_rows = shape[0];
        int A_cols = shape[1];
        int B_rows = B->shape[0];
        int B_cols = B->shape[1];

        if (A_cols != B_rows) throw std::invalid_argument("Matrix dimensions not aligned");

        std::vector<double> C_data(A_rows * B_cols, 0.0);
        for (int i = 0; i < A_rows; ++i) {
            for (int j = 0; j < B_cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < A_cols; ++k) {
                    sum += data[get_index(i, k)] * B->data[B->get_index(k, j)];
                }
                C_data[(i * B_cols) + j] = sum;
            }
        }

        auto C = std::make_shared<Tensor>(A_rows, B_cols, C_data,
                        std::vector<std::shared_ptr<Tensor>>{shared_from_this(), B});

        C->_backward = [C, A = shared_from_this(), B]() {
            backward_matmul_dA(*A, *B, *C);
            backward_matmul_dB(*A, *B, *C);
        };

        return C;
    }

    void build_topo(std::shared_ptr<Tensor> node,
                    std::unordered_set<Tensor*>& visited,
                    std::vector<std::shared_ptr<Tensor>>& topo) {
        if (visited.find(node.get()) == visited.end()) {
            visited.insert(node.get());

            // DFS over computation graph
            for (auto& child : node->_prev) {
                build_topo(child, visited, topo);
            }

            topo.push_back(node);
        }
    }

    void backward() {
        std::vector<std::shared_ptr<Tensor>> topo;
        std::unordered_set<Tensor*> visited;

        build_topo(shared_from_this(), visited, topo);

        this->grad[0] = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }
};
