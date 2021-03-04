cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(texgen LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)


add_executable(texgen src/waves.cu)
target_compile_features(texgen PUBLIC cxx_std_14)
target_link_libraries(texgen ${OpenCV_LIBS})
set_property(TARGET texgen
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET texgen PROPERTY CUDA_ARCHITECTURES 61)

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET texgen
        PROPERTY
        BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
