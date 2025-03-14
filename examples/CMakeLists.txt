cmake_minimum_required(VERSION 3.10)
project(onnxruntime_example)

set(CMAKE_CXX_STANDARD 17)

# Set the path to your ONNX Runtime installation.
# For example, if you unzipped ONNX Runtime to:
# /mnt/c/Machine Learning/Rat Tracking using TensorFlow/onnxruntime
set(ONNXRUNTIME_ROOT "./onnxruntime-linux-x64-1.21.0")

# Include ONNX Runtime headers and libraries.
include_directories("${ONNXRUNTIME_ROOT}/include")
link_directories("${ONNXRUNTIME_ROOT}/lib")

# (Optional) Find OpenCV if needed.
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Add the executable from main.cpp.
add_executable(example-app main.cpp)

# Link the ONNX Runtime library (and OpenCV libraries, if needed).
target_link_libraries(example-app PRIVATE onnxruntime ${OpenCV_LIBS})
