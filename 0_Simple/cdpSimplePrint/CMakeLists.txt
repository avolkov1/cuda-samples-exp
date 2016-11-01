# Using CMake to compile the cdpSimplePrint example with dynamic parallelism.
cmake_minimum_required(VERSION 3.6)

project(cdpSimplePrint)

# If found will print location of the CUDA Toolkit
find_package(CUDA REQUIRED)

# FindCUDA.cmake invoked by find_package(CUDA REQUIRED) defines the following
# functions/macros/variables used below:
#   cuda_find_library_local_first, CUDA_VERSION, CUDA_NVCC_FLAGS

# The cudadevrt library is fixed in newer versions of cmake.
# https://gitlab.kitware.com/cmake/cmake/commit/5717121e81236ba43c87b53f936a8731204bd88d?view=inline
if (NOT CUDA_cudadevrt_LIBRARY) # backport CUDA_cudadevrt_LIBRARY
    cuda_find_library_local_first(CUDA_cudadevrt_LIBRARY
        cudadevrt "\"cudadevrt\" library")
endif(NOT CUDA_cudadevrt_LIBRARY)

# host compiler
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CUDA_VERBOSE_BUILD ON)
set(CUDA_SEPARABLE_COMPILATION ON)

# Common includes and paths for CUDA
set(INCLUDES  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/inc)

################################################################################

# Gencode arguments
if (NOT DEFINED ENV{SMS})
    set(SMS 35 37 50)
    if((CUDA_VERSION VERSION_GREATER "7.0") OR
            (CUDA_VERSION VERSION_EQUAL "7.0"))
        list(APPEND SMS 52) # for cuda cuda 7.0, 7.5
    endif()
    if((CUDA_VERSION VERSION_GREATER "8.0") OR
            (CUDA_VERSION VERSION_EQUAL "8.0"))
        list(APPEND SMS 60) # for cuda 8.0
    endif()
endif (NOT DEFINED ENV{SMS})

unset(GENCODE_FLAGS)
foreach(sm ${SMS})
    list(APPEND GENCODE_FLAGS "-gencode=arch=compute_${sm},code=sm_${sm}")
endforeach()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; ${GENCODE_FLAGS})

################################################################################

# Target rules

# If the .c/.cpp code is separate from .cu use CUDA_SOURCE_PROPERTY_FORMAT.
# set_source_files_properties(cdpSimplePrint.cpp
#     PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

set(SOURCES cdpSimplePrint.cu)

include_directories(${INCLUDES})

# create executable target
cuda_add_executable(cdpSimplePrint ${SOURCES})

#link host with cuda libs
target_link_libraries(cdpSimplePrint
    ${CUDA_cudadevrt_LIBRARY})
