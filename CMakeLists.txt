cmake_minimum_required(VERSION 3.16)
project(linear_regression)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY   ${PROJECT_SOURCE_DIR}/build)
add_executable(linear_regression src/main.cpp)