cmake_minimum_required(VERSION 3.10)
project(RDP_Project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path to ONNX Runtime
set(ONNX_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.21.1)

# Include directories
include_directories(${ONNX_RUNTIME_DIR}/include)

# Link directories
link_directories(${ONNX_RUNTIME_DIR}/lib)

# Add executables
add_executable(helloworld helloworld.cpp)

add_executable(tester tester.cpp)

add_executable(PSNN PSNN.cpp)
target_link_libraries(PSNN onnxruntime)

# Set output directory for all targets
set_target_properties(helloworld tester PSNN
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")