#include "mllama/tensor.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace mllama {
namespace detail {

const char* callerName(const char* caller) {
    return caller == nullptr ? "Tensor" : caller;
}

std::string shapeToString(const std::vector<int>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

// 需要保证shape均通过此校验
size_t checkedNumel(const Tensor& t, const char* caller) {
    size_t total = 1;
    for (size_t axis = 0; axis < t.shape.size(); ++axis) {
        const int dim = t.shape[axis];
        if (dim < 1) {
            throw std::runtime_error(std::string(callerName(caller)) + ": dimension at axis " + std::to_string(axis) +
                                     " must be great, got " + std::to_string(dim) + " in shape " +
                                     shapeToString(t.shape));
        }
        // 预测 total * kDimSize 会不会溢出
        const size_t kDimSize = static_cast<size_t>(dim);
        if (total > std::numeric_limits<size_t>::max() / kDimSize) {
            throw std::runtime_error(std::string(callerName(caller)) + ": shape element count overflow for " +
                                     shapeToString(t.shape));
        }
        total *= kDimSize;
    }
    return total;
}

void checkAxisIndex(const Tensor& t, int axis, int flat_index, const char* caller) {
    const int dim = t.shape[axis];
    if (flat_index < 0 || flat_index >= dim) {
        throw std::out_of_range(std::string(caller) + ": FlatIndex " + std::to_string(flat_index) +
                                " out of range for axis " + std::to_string(axis) + " with size " + std::to_string(dim) +
                                " in tensor's shape: " + shapeToString(t.shape));
    }
}

void checkRank(const Tensor& t, int expected, const char* caller) {
    if (t.dimsNum() != static_cast<size_t>(expected)) {
        throw std::runtime_error(std::string(caller) + " expected " + std::to_string(expected) + "D tensor, got " +
                                 shapeToString(t.shape));
    }
}

}  // namespace detail

// 构造时抛出异常，只要保证构造函数不做malloc等裸资源操作，都能自动回滚
Tensor::Tensor(const std::vector<int>& input_shape, float fill) : shape(input_shape) {
    const size_t kTotal = detail::checkedNumel(*this, "Tensor constructor");
    data.resize(kTotal, fill);
}

size_t Tensor::flatIndex(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::runtime_error("Tensor::flatIndex expected " + std::to_string(shape.size()) + " indices for tensor " +
                                 detail::shapeToString(shape) + ", got " + std::to_string(indices.size()));
    }
    for (size_t axis = 0; axis < indices.size(); ++axis) {
        detail::checkAxisIndex(*this, static_cast<int>(axis), indices[axis], "Tensor::flatIndex");
    }

    size_t flat = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        flat += static_cast<size_t>(indices[i]) * stride;
        stride *= static_cast<size_t>(shape[i]);
    }
    return flat;
}

float& Tensor::at(const std::vector<int>& indices) {
    return data[flatIndex(indices)];
}
float Tensor::at(const std::vector<int>& indices) const {
    return data[flatIndex(indices)];
}

float& Tensor::at1(int d0) {
    detail::checkRank(*this, 1, "Tensor::at1");
    detail::checkAxisIndex(*this, 0, d0, "Tensor::at1");
    return data[d0];
}
float Tensor::at1(int d0) const {
    detail::checkRank(*this, 1, "Tensor::at1");
    detail::checkAxisIndex(*this, 0, d0, "Tensor::at1");
    return data[d0];
}

float& Tensor::at2(int d0, int d1) {
    detail::checkRank(*this, 2, "Tensor::at2");
    detail::checkAxisIndex(*this, 0, d0, "Tensor::at2");
    detail::checkAxisIndex(*this, 1, d1, "Tensor::at2");
    return data[d0 * shape[1] + d1];
}
float Tensor::at2(int d0, int d1) const {
    detail::checkRank(*this, 2, "Tensor::at2");
    detail::checkAxisIndex(*this, 0, d0, "Tensor::at2");
    detail::checkAxisIndex(*this, 1, d1, "Tensor::at2");
    return data[d0 * shape[1] + d1];
}

float& Tensor::at3(int d0, int d1, int d2) {
    detail::checkRank(*this, 3, "Tensor::at3");
    detail::checkAxisIndex(*this, 0, d0, "Tensor::at3");
    detail::checkAxisIndex(*this, 1, d1, "Tensor::at3");
    detail::checkAxisIndex(*this, 2, d2, "Tensor::at3");
    return data[(d0 * shape[1] + d1) * shape[2] + d2];
}
float Tensor::at3(int d0, int d1, int d2) const {
    detail::checkRank(*this, 3, "Tensor::at3");
    detail::checkAxisIndex(*this, 0, d0, "Tensor::at3");
    detail::checkAxisIndex(*this, 1, d1, "Tensor::at3");
    detail::checkAxisIndex(*this, 2, d2, "Tensor::at3");
    return data[(d0 * shape[1] + d1) * shape[2] + d2];
}

float& Tensor::at4(int d0, int d1, int d2, int d3) {
    detail::checkRank(*this, 4, "Tensor::at4");
    detail::checkAxisIndex(*this, 0, d0, "Tensor::at4");
    detail::checkAxisIndex(*this, 1, d1, "Tensor::at4");
    detail::checkAxisIndex(*this, 2, d2, "Tensor::at4");
    detail::checkAxisIndex(*this, 3, d3, "Tensor::at4");
    return data[((d0 * shape[1] + d1) * shape[2] + d2)*shape[3]+d3];
}
float Tensor::at4(int d0, int d1, int d2, int d3) const {
    detail::checkRank(*this, 4, "Tensor::at4");
    detail::checkAxisIndex(*this, 0, d0, "Tensor::at4");
    detail::checkAxisIndex(*this, 1, d1, "Tensor::at4");
    detail::checkAxisIndex(*this, 2, d2, "Tensor::at4");
    detail::checkAxisIndex(*this, 3, d3, "Tensor::at4");
    return data[((d0 * shape[1] + d1) * shape[2] + d2)*shape[3]+d3];
}
// Create a 1D tensor
Tensor makeTensor1D(int d0, float fill) {
    return Tensor({d0}, fill);
}

// Create a 2D tensor
Tensor makeTensor2D(int d0, int d1, float fill) {
    return Tensor({d0, d1}, fill);
}

// Create a 3D tensor
Tensor makeTensor3D(int d0, int d1, int d2, float fill) {
    return Tensor({d0, d1, d2}, fill);
}

// Create a 4D tensor
Tensor makeTensor4D(int d0, int d1, int d2, int d3, float fill) {
    return Tensor({d0, d1, d2, d3}, fill);
}

}  // namespace mllama