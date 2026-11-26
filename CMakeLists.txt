cmake_minimum_required(VERSION 3.14)

project(mini_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 自动收集源码
file(GLOB ML_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

add_library(mllama_core OBJECT ${ML_SOURCES})
target_include_directories(mllama_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

foreach(lib_type STATIC SHARED)
    add_library(mllama_${lib_type} ${lib_type}
        $<TARGET_OBJECTS:mllama_core>
    )
    target_include_directories(mllama_${lib_type} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    set_target_properties(mllama_${lib_type} PROPERTIES
        OUTPUT_NAME mllama
    )
endforeach()

add_library(mllama::mllama ALIAS mllama_SHARED)

# ---- 单元测试（gtest）
enable_testing()
find_package(GTest REQUIRED)
add_executable(tensor_test tests/tensor_test.cc)
target_link_libraries(tensor_test PRIVATE mllama_core GTest::gtest GTest::gtest_main)
target_compile_options(tensor_test PRIVATE -Wall -Wextra)
include(GoogleTest)
gtest_discover_tests(tensor_test)
