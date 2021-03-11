cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(texgen LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

add_library(texgen STATIC
    src/hash.cu
    src/perlin.cu)
target_include_directories(texgen PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(texgen PUBLIC cxx_std_14)
set_property(TARGET texgen PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET texgen PROPERTY CUDA_ARCHITECTURES 61)

add_executable(texgen_test src/test.cu)
target_compile_features(texgen_test PUBLIC cxx_std_14)
target_link_libraries(texgen_test ${OpenCV_LIBS} texgen)
set_property(TARGET texgen_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET texgen_test PROPERTY CUDA_ARCHITECTURES 61)
