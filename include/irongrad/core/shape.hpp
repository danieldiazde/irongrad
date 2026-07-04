#pragma once

#include <cstddef>
#include <stdexcept>

namespace irongrad {

class Shape {
public:
    Shape(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }
    }

    static Shape vector(std::size_t length) {
        return Shape(1, length);
    }

    std::size_t rows() const {
        return rows_;
    }

    std::size_t cols() const {
        return cols_;
    }

    std::size_t size() const {
        return rows_ * cols_;
    }

    bool operator==(const Shape& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_;
    }

    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

private:
    std::size_t rows_;
    std::size_t cols_;
};

} // namespace irongrad
