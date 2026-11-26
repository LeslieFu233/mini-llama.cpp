#pragma once

#include "mllama/tensor.h"
#include "mllama/ops_impl/matmul_impl.h"

namespace mllama {
// Matrix multiplication: c = a * b
// a: [m, k], b: [k, n], c: [m, n]
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);
}  // namespace mllama