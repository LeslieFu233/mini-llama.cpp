// tests/tensor_test.cc —— mllama::Tensor（include/mllama/tensor.h）的 gtest 用例。
//
// 覆盖接口：Tensor(shape, fill) 构造、totalShapeNum / dimsNum、flatIndex、
// at / at1 / at2 / at3 / at4、getRowPtr、makeTensor1D ~ makeTensor4D。
//
// 断言原则：
//   1. 用例断言的是"正确语义"（行主序展开、全下标校验、const 版本返回元素）；
//      如果实现与用例不符，测试会失败——失败信息就是待修的 bug 提示。
//   2. 异常约定取自当前实现：秩不匹配/非法维度 -> std::runtime_error，
//      下标越界 -> std::out_of_range。
//   3. 文件末尾的 TensorAt4Test 三个用例按完整语义断言（含 d3 维），用于检验
//      at4 的两个重载是否写完整。
//
// 运行：cmake --build build && ctest --test-dir build --output-on-failure

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "mllama/tensor.h"

namespace mllama {
namespace {

// 生成 data[i] = i 的张量，便于把"下标算错"直接暴露成"取值不对"。
Tensor makeFilled(const std::vector<int>& shape) {
    Tensor t(shape);
    for (size_t i = 0; i < t.data.size(); ++i) {
        t.data[i] = static_cast<float>(i);
    }
    return t;
}

// ========================== 构造 ==========================

TEST(TensorConstructTest, DefaultTensorIsEmpty) {
    const Tensor t;

    EXPECT_TRUE(t.data.empty());
    EXPECT_TRUE(t.shape.empty());
    EXPECT_EQ(t.dimsNum(), 0U);
}

TEST(TensorConstructTest, ShapeConstructorAllocatesFilledTensor) {
    const Tensor t({2, 3}, 1.5f);

    EXPECT_EQ(t.shape, std::vector<int>({2, 3}));
    EXPECT_EQ(t.dimsNum(), 2U);
    EXPECT_EQ(t.data.size(), 6U);
    for (float value : t.data) {
        EXPECT_FLOAT_EQ(value, 1.5f);
    }
}

TEST(TensorConstructTest, ShapeConstructorDefaultsToZeroFill) {
    const Tensor t({2, 2});

    EXPECT_EQ(t.data.size(), 4U);
    for (float value : t.data) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(TensorConstructTest, ConstructorSupports1DTo4D) {
    Tensor t1({5});
    EXPECT_EQ(t1.dimsNum(), 1U);
    EXPECT_EQ(t1.totalShapeNum(), 5U);

    Tensor t2({2, 3});
    EXPECT_EQ(t2.dimsNum(), 2U);
    EXPECT_EQ(t2.totalShapeNum(), 6U);

    Tensor t3({2, 3, 4});
    EXPECT_EQ(t3.dimsNum(), 3U);
    EXPECT_EQ(t3.totalShapeNum(), 24U);

    Tensor t4({2, 3, 4, 5});
    EXPECT_EQ(t4.dimsNum(), 4U);
    EXPECT_EQ(t4.totalShapeNum(), 120U);
    EXPECT_EQ(t4.data.size(), 120U);
}

TEST(TensorConstructTest, ConstructorThrowsOnDimLessThanOne) {
    EXPECT_THROW(Tensor({2, 0}), std::runtime_error);
    EXPECT_THROW(Tensor({-1, 3}), std::runtime_error);
}

TEST(TensorConstructTest, ConstructorThrowsOnElementCountOverflow) {
    // 2^22 * 2^22 * 2^22 = 2^66，超出 size_t 范围
    EXPECT_THROW(Tensor({1 << 22, 1 << 22, 1 << 22}), std::runtime_error);
}

// ========================== 查询 ==========================

TEST(TensorQueryTest, DimsNumMatchesShapeSize) {
    const Tensor t1({7});
    EXPECT_EQ(t1.dimsNum(), 1U);

    const Tensor t2({2, 3});
    EXPECT_EQ(t2.dimsNum(), 2U);

    const Tensor t3({2, 3, 4});
    EXPECT_EQ(t3.dimsNum(), 3U);
}

TEST(TensorQueryTest, TotalShapeNumIsProductOfDims) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.totalShapeNum(), 24U);
    EXPECT_EQ(t.totalShapeNum(), t.data.size());
}

// ========================== flatIndex ==========================

TEST(TensorFlatIndexTest, FlatIndexComputesRowMajorOrder) {
    const Tensor t2 = makeFilled({2, 3});
    EXPECT_EQ(t2.flatIndex({0, 0}), 0U);
    EXPECT_EQ(t2.flatIndex({0, 2}), 2U);
    EXPECT_EQ(t2.flatIndex({1, 0}), 3U);
    EXPECT_EQ(t2.flatIndex({1, 2}), 5U);

    const Tensor t3 = makeFilled({2, 3, 4});
    EXPECT_EQ(t3.flatIndex({1, 2, 3}), 23U);  // (1*3+2)*4+3

    const Tensor t4 = makeFilled({2, 3, 4, 5});
    EXPECT_EQ(t4.flatIndex({1, 2, 3, 4}), 119U);  // ((1*3+2)*4+3)*5+4
}

TEST(TensorFlatIndexTest, FlatIndexThrowsOnRankMismatch) {
    const Tensor t2({2, 3});

    EXPECT_THROW(t2.flatIndex({0}), std::runtime_error);
    EXPECT_THROW(t2.flatIndex({0, 0, 0}), std::runtime_error);
}

TEST(TensorFlatIndexTest, FlatIndexThrowsOnIndexOutOfRange) {
    const Tensor t2({2, 3});

    EXPECT_THROW(t2.flatIndex({2, 0}), std::out_of_range);   // 第 0 维越界
    EXPECT_THROW(t2.flatIndex({0, 3}), std::out_of_range);   // 第 1 维越界
    EXPECT_THROW(t2.flatIndex({0, -1}), std::out_of_range);  // 负下标
}

// ========================== at（vector 版本） ==========================

TEST(TensorAtTest, AtReadsAndWritesElements) {
    Tensor t = makeTensor2D(2, 3, 0.0f);

    t.at({0, 1}) = 2.5f;
    t.at({1, 2}) = -4.0f;

    EXPECT_FLOAT_EQ(t.at2(0, 1), 2.5f);
    EXPECT_FLOAT_EQ(t.at2(1, 2), -4.0f);
    EXPECT_FLOAT_EQ(t.data[t.flatIndex({1, 2})], -4.0f);
}

TEST(TensorAtTest, ConstAtReturnsElementValue) {
    const Tensor t = makeTensor2D(2, 3, 1.25f);

    EXPECT_FLOAT_EQ(t.at({1, 2}), 1.25f);
}

TEST(TensorAtTest, AtThrowsOnRankMismatch) {
    const Tensor t2({2, 3});

    EXPECT_THROW(t2.at({0}), std::runtime_error);
    EXPECT_THROW(t2.at({0, 0, 0}), std::runtime_error);
}

TEST(TensorAtTest, AtThrowsOnIndexOutOfRange) {
    const Tensor t2({2, 3});

    EXPECT_THROW(t2.at({2, 0}), std::out_of_range);
    EXPECT_THROW(t2.at({0, -1}), std::out_of_range);
}

// ========================== at1 ~ at3（定秩访问） ==========================

TEST(TensorAtFixedRankTest, At1ReadsAndWrites1D) {
    Tensor t = makeTensor1D(3, 0.0f);

    t.at1(1) = 5.0f;

    EXPECT_FLOAT_EQ(t.at1(1), 5.0f);
    EXPECT_FLOAT_EQ(t.data[1], 5.0f);

    const Tensor& ct = t;
    EXPECT_FLOAT_EQ(ct.at1(1), 5.0f);
}

TEST(TensorAtFixedRankTest, At2MatchesRowMajorOrder) {
    Tensor t = makeFilled({2, 3});

    EXPECT_FLOAT_EQ(t.at2(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t.at2(0, 2), 2.0f);
    EXPECT_FLOAT_EQ(t.at2(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(t.at2(1, 2), 5.0f);  // d0 * shape[1] + d1

    t.at2(1, 2) = 9.0f;
    EXPECT_FLOAT_EQ(t.at({1, 2}), 9.0f);  // 与 at(vector) 一致
}

TEST(TensorAtFixedRankTest, At3MatchesRowMajorOrder) {
    const Tensor t = makeFilled({2, 3, 4});

    EXPECT_FLOAT_EQ(t.at3(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t.at3(1, 2, 3), 23.0f);  // (d0*shape[1]+d1)*shape[2]+d2
    EXPECT_FLOAT_EQ(t.at3(0, 1, 2), 6.0f);
}

TEST(TensorAtFixedRankTest, FixedRankAccessorsThrowOnWrongRank) {
    const Tensor t2({2, 3});
    EXPECT_THROW(t2.at1(0), std::runtime_error);
    EXPECT_THROW(t2.at3(0, 0, 0), std::runtime_error);
    EXPECT_THROW(t2.at4(0, 0, 0, 0), std::runtime_error);

    const Tensor t3({2, 3, 4});
    EXPECT_THROW(t3.at2(0, 0), std::runtime_error);
    EXPECT_THROW(t3.at4(0, 0, 0, 0), std::runtime_error);
}

TEST(TensorAtFixedRankTest, At1ToAt3ThrowOnIndexOutOfRange) {
    const Tensor t1({3});
    EXPECT_THROW(t1.at1(3), std::out_of_range);
    EXPECT_THROW(t1.at1(-1), std::out_of_range);

    const Tensor t2({2, 3});
    EXPECT_THROW(t2.at2(2, 0), std::out_of_range);
    EXPECT_THROW(t2.at2(0, 3), std::out_of_range);

    const Tensor t3({2, 3, 4});
    EXPECT_THROW(t3.at3(0, 0, 4), std::out_of_range);
    EXPECT_THROW(t3.at3(0, -1, 0), std::out_of_range);
}

// ========================== getRowPtr ==========================

TEST(TensorGetRowPtrTest, GetRowPtrPointsToRowStart) {
    Tensor t = makeTensor2D(3, 4, 0.0f);

    EXPECT_EQ(t.getRowPtr(0), t.data.data());
    EXPECT_EQ(t.getRowPtr(1), t.data.data() + 4);  // 行步长 = shape[1]
    EXPECT_EQ(t.getRowPtr(2), t.data.data() + 8);
}

TEST(TensorGetRowPtrTest, GetRowPtrRowSliceMatchesAt2) {
    Tensor t = makeTensor2D(3, 4, 0.0f);

    t.at2(1, 2) = 7.5f;

    EXPECT_FLOAT_EQ(t.getRowPtr(1)[2], 7.5f);
    EXPECT_FLOAT_EQ(t.getRowPtr(1)[0], t.at2(1, 0));
}

// ========================== makeTensorXD 工厂 ==========================

TEST(TensorFactoryTest, MakeTensor1DCreatesFilledVector) {
    // 注意：totalShapeNum() 当前不是 const 成员，这里只能用非 const 对象调用
    Tensor t = makeTensor1D(5, 0.5f);

    EXPECT_EQ(t.dimsNum(), 1U);
    EXPECT_EQ(t.totalShapeNum(), 5U);
    EXPECT_EQ(t.data.size(), 5U);
    for (float value : t.data) {
        EXPECT_FLOAT_EQ(value, 0.5f);
    }
}

TEST(TensorFactoryTest, MakeTensor2DCreatesFilledMatrix) {
    Tensor t = makeTensor2D(2, 3, -1.0f);

    EXPECT_EQ(t.shape, std::vector<int>({2, 3}));
    EXPECT_EQ(t.totalShapeNum(), 6U);
    EXPECT_FLOAT_EQ(t.at2(1, 2), -1.0f);
}

TEST(TensorFactoryTest, MakeTensor3DCreatesFilledCube) {
    Tensor t = makeTensor3D(2, 3, 4, 2.0f);

    EXPECT_EQ(t.dimsNum(), 3U);
    EXPECT_EQ(t.totalShapeNum(), 24U);
    EXPECT_FLOAT_EQ(t.at3(1, 2, 3), 2.0f);
}

TEST(TensorFactoryTest, MakeTensor4DCreatesFilledTensor) {
    Tensor t = makeTensor4D(2, 3, 4, 5, 0.25f);

    EXPECT_EQ(t.dimsNum(), 4U);
    EXPECT_EQ(t.totalShapeNum(), 120U);
    EXPECT_EQ(t.data.size(), 120U);
    EXPECT_FLOAT_EQ(t.data[t.flatIndex({1, 2, 3, 4})], 0.25f);
}

// ============ at4 专项（按完整语义断言：下标含 d3、d3 须校验） ============
// 以下 3 个用例检验 at4 的两个重载是否实现完整；若失败，请对照
// at2 / at3 的写法补齐（对照组：at3 的公式与校验是完整的）。

TEST(TensorAt4Test, At4MatchesRowMajorOrder) {
    Tensor t = makeFilled({2, 3, 4, 5});

    // 期望 ((d0*shape[1]+d1)*shape[2]+d2)*shape[3]+d3
    EXPECT_FLOAT_EQ(t.at4(0, 0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t.at4(1, 2, 3, 0), 115.0f);
    EXPECT_FLOAT_EQ(t.at4(1, 2, 3, 4), 119.0f);
    EXPECT_FLOAT_EQ(t.at4(1, 2, 3, 4), t.at({1, 2, 3, 4}));  // 与 at 一致

    t.at4(1, 2, 3, 4) = 999.0f;  // 可写，且必须写进 data[119]
    EXPECT_FLOAT_EQ(t.data[119], 999.0f);
}

TEST(TensorAt4Test, At4ThrowsOnIndexOutOfRange) {
    const Tensor t = makeTensor4D(2, 3, 4, 5, 0.0f);

    // 边界正例：每一维的最后合法下标都能正常访问，不抛异常
    EXPECT_NO_THROW(t.at4(1, 2, 3, 4));
    EXPECT_NO_THROW(t.at4(1, 0, 0, 0));
    // 最后一维同样必须校验
    EXPECT_THROW(t.at4(0, 0, 0, 5), std::out_of_range);
    EXPECT_THROW(t.at4(0, 0, 0, -1), std::out_of_range);
    EXPECT_THROW(t.at4(2, 0, 0, 0), std::out_of_range);
}

TEST(TensorAt4Test, ConstAt4ReturnsElementValue) {
    const Tensor t = makeFilled({2, 3, 4, 5});

    EXPECT_FLOAT_EQ(t.at4(1, 2, 3, 4), 119.0f);
}

}  // namespace
}  // namespace mllama
