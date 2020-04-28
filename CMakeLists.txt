cmake_minimum_required(VERSION 3.16)
project(linear_regression)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY   ${PROJECT_SOURCE_DIR}/build)

find_package(Python COMPONENTS Development NumPy)

message("Python_VERSION:${Python_VERSION}")
message("Python_NumPy_VERSION:${Python_NumPy_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")
message("Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS}")
message("Python_NumPy_INCLUDE_DIRS:${Python_NumPy_INCLUDE_DIRS}")

add_executable(linear_regression src/main.cpp src/CSV.cpp src/CSV.h src/Data.cpp src/Data.h src/matplotlib.h)

target_include_directories(linear_regression PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(linear_regression ${Python_LIBRARIES})