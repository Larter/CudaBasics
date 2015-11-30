cmake_minimum_required(VERSION 2.8)
project(CudaProject)
find_package(CUDA REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++11)
#INCLUDE_DIRECTORIES(/usr/local/cuda/include)
cuda_add_executable(hellocuda main.cu)



file(GLOB EX1_SRCS ./*.cu)

cuda_add_executable(simpleThrust ${EX1_SRCS})



#add_executable(CudaProject ${SOURCE_FILES})
