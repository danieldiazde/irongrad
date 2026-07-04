#include "irongrad/core/tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace irongrad {

Tensor::Tensor(std::vector<double> data)
    : shape_(Shape::vector(data.size())),
      data_(std::move(data)),
      grad_(data_.size(), 0.0),
      backward_fn_([](){}) {
    validate_storage();
}

Tensor::Tensor(Shape shape, std::vector<double> data)
    : shape_(shape),
      data_(std::move(data)),
      grad_(data_.size(), 0.0),
      backward_fn_([](){}) {
    validate_storage();
}

Tensor::Tensor(Shape shape, std::vector<double> data, std::vector<Ptr> parents)
    : shape_(shape),
      data_(std::move(data)),
      grad_(data_.size(), 0.0),
      parents_(std::move(parents)),
      backward_fn_([](){}) {
    validate_storage();
}

Tensor::Ptr Tensor::create(std::vector<double> data) {
    return std::make_shared<Tensor>(std::move(data));
}

Tensor::Ptr Tensor::create(Shape shape, std::vector<double> data) {
    return std::make_shared<Tensor>(shape, std::move(data));
}

const Shape& Tensor::shape() const {
    return shape_;
}

std::size_t Tensor::rows() const {
    return shape_.rows();
}

std::size_t Tensor::cols() const {
    return shape_.cols();
}

std::size_t Tensor::size() const {
    return data_.size();
}

std::size_t Tensor::index(std::size_t row, std::size_t col) const {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("Tensor index out of range");
    }

    return (row * cols()) + col;
}

const std::vector<double>& Tensor::data() const {
    return data_;
}

const std::vector<double>& Tensor::grad() const {
    return grad_;
}

std::vector<double>& Tensor::mutable_data() {
    return data_;
}

std::vector<double>& Tensor::mutable_grad() {
    return grad_;
}

void Tensor::set_data(std::vector<double> data) {
    if (data.size() != shape_.size()) {
        throw std::invalid_argument("Data size must match tensor shape");
    }

    data_ = std::move(data);
}

void Tensor::set_grad(std::vector<double> grad) {
    if (grad.size() != shape_.size()) {
        throw std::invalid_argument("Gradient size must match tensor shape");
    }

    grad_ = std::move(grad);
}

void Tensor::zero_grad() {
    std::fill(grad_.begin(), grad_.end(), 0.0);
}

Tensor::Ptr Tensor::add(const Ptr& other) {
    require_tensor(other);
    require_same_shape(*other);

    std::vector<double> output(size());
    for (std::size_t i = 0; i < size(); ++i) {
        output[i] = data_[i] + other->data_[i];
    }

    auto self = shared_from_this();
    auto out = std::make_shared<Tensor>(shape_, std::move(output), std::vector<Ptr>{self, other});
    out->backward_fn_ = [out, self, other]() {
        for (std::size_t i = 0; i < out->grad_.size(); ++i) {
            self->grad_[i] += out->grad_[i];
            other->grad_[i] += out->grad_[i];
        }
    };

    return out;
}

Tensor::Ptr Tensor::add_row_vector(const Ptr& row_vector) {
    require_tensor(row_vector);
    require_row_vector_for_broadcast(*row_vector);

    std::vector<double> output(size());
    for (std::size_t row = 0; row < rows(); ++row) {
        for (std::size_t col = 0; col < cols(); ++col) {
            output[index(row, col)] = data_[index(row, col)] + row_vector->data_[col];
        }
    }

    auto self = shared_from_this();
    auto out = std::make_shared<Tensor>(shape_, std::move(output), std::vector<Ptr>{self, row_vector});
    out->backward_fn_ = [out, self, row_vector]() {
        for (std::size_t row = 0; row < self->rows(); ++row) {
            for (std::size_t col = 0; col < self->cols(); ++col) {
                const double upstream = out->grad_[out->index(row, col)];
                self->grad_[self->index(row, col)] += upstream;
                row_vector->grad_[col] += upstream;
            }
        }
    };

    return out;
}

Tensor::Ptr Tensor::mul(const Ptr& other) {
    require_tensor(other);
    require_same_shape(*other);

    std::vector<double> output(size());
    for (std::size_t i = 0; i < size(); ++i) {
        output[i] = data_[i] * other->data_[i];
    }

    auto self = shared_from_this();
    auto out = std::make_shared<Tensor>(shape_, std::move(output), std::vector<Ptr>{self, other});
    out->backward_fn_ = [out, self, other]() {
        for (std::size_t i = 0; i < out->grad_.size(); ++i) {
            self->grad_[i] += other->data_[i] * out->grad_[i];
            other->grad_[i] += self->data_[i] * out->grad_[i];
        }
    };

    return out;
}

Tensor::Ptr Tensor::relu() {
    std::vector<double> output(size());
    for (std::size_t i = 0; i < size(); ++i) {
        output[i] = data_[i] > 0.0 ? data_[i] : 0.0;
    }

    auto self = shared_from_this();
    auto out = std::make_shared<Tensor>(shape_, std::move(output), std::vector<Ptr>{self});
    out->backward_fn_ = [out, self]() {
        for (std::size_t i = 0; i < out->grad_.size(); ++i) {
            self->grad_[i] += (self->data_[i] > 0.0 ? 1.0 : 0.0) * out->grad_[i];
        }
    };

    return out;
}

Tensor::Ptr Tensor::sum() {
    double total = 0.0;
    for (double value : data_) {
        total += value;
    }

    auto self = shared_from_this();
    auto out = std::make_shared<Tensor>(Shape(1, 1), std::vector<double>{total}, std::vector<Ptr>{self});
    out->backward_fn_ = [out, self]() {
        for (double& value : self->grad_) {
            value += out->grad_[0];
        }
    };

    return out;
}

Tensor::Ptr Tensor::matmul(const Ptr& other) {
    return matmul_impl(other, MatmulKind::RowMajor);
}

Tensor::Ptr Tensor::matmul_naive(const Ptr& other) {
    return matmul_impl(other, MatmulKind::Naive);
}

Tensor::Ptr Tensor::matmul_tiled(const Ptr& other) {
    return matmul_impl(other, MatmulKind::Tiled);
}

Tensor::Ptr Tensor::matmul_impl(const Ptr& other, MatmulKind kind) {
    require_tensor(other);

    if (cols() != other->rows()) {
        throw std::invalid_argument("Matrix dimensions not aligned");
    }

    const std::size_t lhs_rows = rows();
    const std::size_t lhs_cols = cols();
    const std::size_t rhs_cols = other->cols();

    std::vector<double> output(lhs_rows * rhs_cols, 0.0);

    switch (kind) {
    case MatmulKind::RowMajor: {
        for (std::size_t row = 0; row < lhs_rows; ++row) {
            const std::size_t lhs_row_offset = row * lhs_cols;
            const std::size_t out_row_offset = row * rhs_cols;

            for (std::size_t k = 0; k < lhs_cols; ++k) {
                const double lhs_value = data_[lhs_row_offset + k];
                const std::size_t rhs_row_offset = k * rhs_cols;

                for (std::size_t col = 0; col < rhs_cols; ++col) {
                    output[out_row_offset + col] += lhs_value * other->data_[rhs_row_offset + col];
                }
            }
        }
        break;
    }
    case MatmulKind::Naive: {
        for (std::size_t row = 0; row < lhs_rows; ++row) {
            const std::size_t lhs_row_offset = row * lhs_cols;
            const std::size_t out_row_offset = row * rhs_cols;

            for (std::size_t col = 0; col < rhs_cols; ++col) {
                double value = 0.0;
                for (std::size_t k = 0; k < lhs_cols; ++k) {
                    value += data_[lhs_row_offset + k] * other->data_[(k * rhs_cols) + col];
                }
                output[out_row_offset + col] = value;
            }
        }
        break;
    }
    case MatmulKind::Tiled: {
        constexpr std::size_t block = 32;
        const double* const lhs_data = data_.data();
        const double* const rhs_data = other->data_.data();
        double* const out_data = output.data();

        for (std::size_t i0 = 0; i0 < lhs_rows; i0 += block) {
            const std::size_t i_end = std::min(i0 + block, lhs_rows);

            for (std::size_t k0 = 0; k0 < lhs_cols; k0 += block) {
                const std::size_t k_end = std::min(k0 + block, lhs_cols);

                for (std::size_t j0 = 0; j0 < rhs_cols; j0 += block) {
                    const std::size_t j_end = std::min(j0 + block, rhs_cols);

                    for (std::size_t i = i0; i < i_end; ++i) {
                        const std::size_t lhs_row_offset = i * lhs_cols;
                        double* const out_row = out_data + (i * rhs_cols);

                        for (std::size_t k = k0; k < k_end; ++k) {
                            const double lhs_value = lhs_data[lhs_row_offset + k];
                            const double* const rhs_row = rhs_data + (k * rhs_cols);

                            for (std::size_t j = j0; j < j_end; ++j) {
                                out_row[j] += lhs_value * rhs_row[j];
                            }
                        }
                    }
                }
            }
        }
        break;
    }
    }

    auto self = shared_from_this();
    auto out = std::make_shared<Tensor>(
        Shape(lhs_rows, rhs_cols),
        std::move(output),
        std::vector<Ptr>{self, other}
    );

    out->backward_fn_ = [out, self, other]() {
        const std::size_t lhs_rows = self->rows();
        const std::size_t lhs_cols = self->cols();
        const std::size_t rhs_cols = other->cols();

        for (std::size_t row = 0; row < lhs_rows; ++row) {
            const std::size_t lhs_row_offset = row * lhs_cols;
            const std::size_t grad_row_offset = row * rhs_cols;

            for (std::size_t col = 0; col < lhs_cols; ++col) {
                double value = 0.0;
                const std::size_t rhs_row_offset = col * rhs_cols;

                for (std::size_t k = 0; k < rhs_cols; ++k) {
                    value += out->grad_[grad_row_offset + k] * other->data_[rhs_row_offset + k];
                }
                self->grad_[lhs_row_offset + col] += value;
            }
        }

        for (std::size_t row = 0; row < lhs_cols; ++row) {
            const std::size_t rhs_row_offset = row * rhs_cols;

            for (std::size_t col = 0; col < rhs_cols; ++col) {
                double value = 0.0;
                for (std::size_t k = 0; k < lhs_rows; ++k) {
                    value += self->data_[(k * lhs_cols) + row] * out->grad_[(k * rhs_cols) + col];
                }
                other->grad_[rhs_row_offset + col] += value;
            }
        }
    };

    return out;
}

void Tensor::backward() {
    std::vector<Ptr> topology;
    std::unordered_set<Tensor*> visited;
    build_topology(shared_from_this(), visited, topology);

    for (const auto& node : topology) {
        if (!node->parents_.empty()) {
            node->zero_grad();
        }
    }

    for (double& value : grad_) {
        value += 1.0;
    }

    for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
        (*it)->backward_fn_();
    }
}

void Tensor::validate_storage() const {
    if (data_.size() != shape_.size()) {
        throw std::invalid_argument("Shape and data size mismatch");
    }
}

void Tensor::require_same_shape(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match");
    }
}

void Tensor::require_row_vector_for_broadcast(const Tensor& other) const {
    if (other.rows() != 1 || other.cols() != cols()) {
        throw std::invalid_argument("Row vector must have shape 1 x tensor columns");
    }
}

void Tensor::require_tensor(const Ptr& tensor) {
    if (!tensor) {
        throw std::invalid_argument("Tensor argument must not be null");
    }
}

void Tensor::build_topology(const Ptr& node,
                            std::unordered_set<Tensor*>& visited,
                            std::vector<Ptr>& topology) {
    if (visited.find(node.get()) != visited.end()) {
        return;
    }

    visited.insert(node.get());
    for (const auto& parent : node->parents_) {
        build_topology(parent, visited, topology);
    }
    topology.push_back(node);
}

} // namespace irongrad
