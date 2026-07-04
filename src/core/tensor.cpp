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
    require_tensor(other);

    if (cols() != other->rows()) {
        throw std::invalid_argument("Matrix dimensions not aligned");
    }

    std::vector<double> output(rows() * other->cols(), 0.0);
    for (std::size_t row = 0; row < rows(); ++row) {
        for (std::size_t col = 0; col < other->cols(); ++col) {
            double value = 0.0;
            for (std::size_t k = 0; k < cols(); ++k) {
                value += data_[index(row, k)] * other->data_[other->index(k, col)];
            }
            output[(row * other->cols()) + col] = value;
        }
    }

    auto self = shared_from_this();
    auto out = std::make_shared<Tensor>(
        Shape(rows(), other->cols()),
        std::move(output),
        std::vector<Ptr>{self, other}
    );

    out->backward_fn_ = [out, self, other]() {
        for (std::size_t row = 0; row < self->rows(); ++row) {
            for (std::size_t col = 0; col < self->cols(); ++col) {
                double value = 0.0;
                for (std::size_t k = 0; k < other->cols(); ++k) {
                    value += out->grad_[out->index(row, k)] * other->data_[other->index(col, k)];
                }
                self->grad_[self->index(row, col)] += value;
            }
        }

        for (std::size_t row = 0; row < other->rows(); ++row) {
            for (std::size_t col = 0; col < other->cols(); ++col) {
                double value = 0.0;
                for (std::size_t k = 0; k < self->rows(); ++k) {
                    value += self->data_[self->index(k, row)] * out->grad_[out->index(k, col)];
                }
                other->grad_[other->index(row, col)] += value;
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
