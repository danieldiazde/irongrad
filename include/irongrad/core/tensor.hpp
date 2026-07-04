#pragma once

#include "irongrad/core/shape.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace irongrad {

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    using Ptr = std::shared_ptr<Tensor>;

    explicit Tensor(std::vector<double> data);
    Tensor(Shape shape, std::vector<double> data);
    Tensor(Shape shape, std::vector<double> data, std::vector<Ptr> parents);

    static Ptr create(std::vector<double> data);
    static Ptr create(Shape shape, std::vector<double> data);

    const Shape& shape() const;
    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t size() const;
    std::size_t index(std::size_t row, std::size_t col) const;

    const std::vector<double>& data() const;
    const std::vector<double>& grad() const;

    std::vector<double>& mutable_data();
    std::vector<double>& mutable_grad();
    void set_data(std::vector<double> data);
    void set_grad(std::vector<double> grad);

    void zero_grad();
    void backward();

    Ptr add(const Ptr& other);
    Ptr add_row_vector(const Ptr& row_vector);
    Ptr mul(const Ptr& other);
    Ptr relu();
    Ptr sum();
    Ptr matmul(const Ptr& other);
    Ptr matmul_naive(const Ptr& other);
    Ptr matmul_tiled(const Ptr& other);

private:
    enum class MatmulKind { Naive, RowMajor, Tiled };

    Shape shape_;
    std::vector<double> data_;
    std::vector<double> grad_;
    std::vector<Ptr> parents_;
    std::function<void()> backward_fn_;

    void validate_storage() const;
    void require_same_shape(const Tensor& other) const;
    void require_row_vector_for_broadcast(const Tensor& other) const;
    Ptr matmul_impl(const Ptr& other, MatmulKind kind);

    static void require_tensor(const Ptr& tensor);
    static void build_topology(const Ptr& node,
                               std::unordered_set<Tensor*>& visited,
                               std::vector<Ptr>& topology);
};

} // namespace irongrad
