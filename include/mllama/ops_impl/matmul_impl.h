#pragma once

namespace mllama {
enum class MatmulMode {
    kNaive,
    kThreaded,
    kSimd,
    kThreadedSimd,
};

MatmulMode defaultMatmulMode();

}