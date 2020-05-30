cmake_minimum_required(VERSION 3.16)
project(ml)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY   ${PROJECT_SOURCE_DIR}/build)

find_package(Python COMPONENTS Development NumPy)
find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED CXX)
find_package(VTK COMPONENTS
        vtkChartsCore
        vtkCommonCore
        vtkCommonColor
        vtkCommonDataModel
        vtkInteractionStyle
        vtkRenderingContext2D
        vtkRenderingContextOpenGL2
        vtkRenderingOpenGL2
        vtkRenderingCore
        vtkRenderingFreeType
        vtkRenderingGL2PSOpenGL2
        vtkViewsContext2D QUIET)
if (NOT VTK_FOUND)
    message("Skipping ScatterPlot: ${VTK_NOT_FOUND_MESSAGE}")
    return ()
endif()

include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG v19.19)
FetchContent_MakeAvailable(dlib)

include(ExternalProject)
ExternalProject_Add(matio
        GIT_REPOSITORY https://github.com/tbeu/matio.git
        GIT_TAG v1.5.17
        PREFIX "_deps/matio"
        SOURCE_DIR ${matio_SOURCE_DIR}
        BUILD_IN_SOURCE 1
        UPDATE_COMMAND ./autogen.sh
        CONFIGURE_COMMAND ./configure)

ExternalProject_Get_Property(matio SOURCE_DIR INSTALL_DIR)
set(matio_SOURCE_DIR ${SOURCE_DIR})
set(matio_INSTALL_DIR ${INSTALL_DIR})

message("Python_VERSION:${Python_VERSION}")
message("Python_NumPy_VERSION:${Python_NumPy_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")
message("Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS}")
message("Python_NumPy_INCLUDE_DIRS:${Python_NumPy_INCLUDE_DIRS}")

message("ZLIB_VERSION_STRING:${ZLIB_VERSION_STRING}")
message("ZLIB_INCLUDE_DIRS:${ZLIB_INCLUDE_DIRS}")
message("ZLIB_LIBRARIES:${ZLIB_LIBRARIES}")

message("HDF5_VERSION:${HDF5_VERSION}")
message("HDF5_CXX_INCLUDE_DIRS:${HDF5_CXX_INCLUDE_DIRS}")

message (STATUS "VTK_VERSION: ${VTK_VERSION}")

message("matio_SOURCE_DIR:${matio_SOURCE_DIR}")

add_executable(ml src/main.cpp include/matplotlibcpp.h src/vtkBetterChartXYZ.h src/vtkBetterChartXYZ.cpp src/ex1a.cpp src/ex1a.h src/ex1b.cpp src/ex1b.h src/utils.h src/utils.cpp src/utils.h src/ex2a.cpp src/ex2a.h src/ex2b.cpp src/ex2b.h src/ex3a.cpp src/ex3a.h)

target_include_directories(ml PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS} /usr/local/include include)
target_link_libraries(ml ${Python_LIBRARIES} dlib::dlib /usr/local/lib/libmatio.dylib ${VTK_LIBRARIES})
vtk_module_autoinit(
        TARGETS ml
        MODULES ${VTK_LIBRARIES}
)
