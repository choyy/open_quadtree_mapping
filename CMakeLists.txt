SET(PROJECT_NAME open_quadtree_mapping)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(${PROJECT_NAME})
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
if(NOT CMAKE_BUILD_TYPE)
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
endif()

find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)

# Specify compute capability
list(APPEND CUDA_NVCC_FLAGS -arch=sm_61)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

cuda_add_library(open_quadtree_mapping SHARED
    src/check_cuda_device.cu
    src/seed_matrix.cu
    src/depthmap.cpp
    # src/main.cpp
    )
target_link_libraries(open_quadtree_mapping
${OpenCV_LIBS} ${Boost_LIBRARIES})

# cuda_add_executable(open_quadtree_mapping_node
#     src/main.cpp
#     )
# target_link_libraries(open_quadtree_mapping_node
# open_quadtree_mapping
# )
