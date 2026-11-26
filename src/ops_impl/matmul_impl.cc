#include "mllama/ops.h"

namespace mllama {

// 需要在这一层之前解决异常，如果已经到具体的计算层了，必须保证维度是正确匹配的

static Tensor matmulNaive(const Tensor& a, const Tensor& b) {
  int out_row = a.shape[0];
  int inner = a.shape[1];
  int out_col = b.shape[1];
  // 输出的 tensor，先创建好，之后填充
  Tensor out({out_row, out_col}, 0.0f);
}
}  // namespace mllama