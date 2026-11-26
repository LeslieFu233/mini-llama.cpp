#pragma once

#include <cstddef>
#include <string>
#include <vector>
namespace mllama {
struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;
    Tensor() = default;
    explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);
    size_t totalShapeNum() const { return data.size(); };
    size_t dimsNum() const { return shape.size(); }
    // 计算展开结果
    size_t flatIndex(const std::vector<int>& indices) const;

    float& at(const std::vector<int>& indices);
    float at(const std::vector<int>& indices) const;

    float& at1(int d0);
    float at1(int d0) const;

    float& at2(int d0, int d1);
    float at2(int d0, int d1) const;

    float& at3(int d0, int d1, int d2);
    float at3(int d0, int d1, int d2) const;

    float& at4(int d0, int d1, int d2, int d3);
    float at4(int d0, int d1, int d2, int d3) const;

    float* getRowPtr(int row) { return &data[row * (totalShapeNum() / static_cast<size_t>(shape[0]))]; }
};

// Create a 1D tensor
Tensor makeTensor1D(int d0, float fill = 0.0f);

// Create a 2D tensor
Tensor makeTensor2D(int d0, int d1, float fill = 0.0f);

// Create a 3D tensor
Tensor makeTensor3D(int d0, int d1, int d2, float fill = 0.0f);

// Create a 4D tensor
Tensor makeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);

}  // namespace mllama