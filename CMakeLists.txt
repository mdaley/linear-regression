cmake_minimum_required(VERSION 3.16)
project(ml)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY   ${PROJECT_SOURCE_DIR}/build)

find_package(Python COMPONENTS Development NumPy)

include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG v19.19
)
FetchContent_MakeAvailable(dlib)

message("Python_VERSION:${Python_VERSION}")
message("Python_NumPy_VERSION:${Python_NumPy_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")
message("Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS}")
message("Python_NumPy_INCLUDE_DIRS:${Python_NumPy_INCLUDE_DIRS}")

add_executable(ml src/main.cpp include/matplotlibcpp.h src/ex1a.cpp src/ex1a.h src/ex1b.cpp src/ex1b.h src/utils.h src/utils.cpp src/utils.h src/ex2a.cpp src/ex2a.h src/ex2b.cpp src/ex2b.h)

target_include_directories(ml PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS} include)
target_link_libraries(ml ${Python_LIBRARIES} dlib::dlib)
