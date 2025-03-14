cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example-app)

# 1. Find libtorch
find_package(Torch REQUIRED)

# 2. Add the TorchVision subdirectory (which you have built or will build)
#    This works if there's a CMakeLists.txt in the 'vision' folder that
#    creates a target like 'torchvision_library'.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vision build_vision)

# 3. Create the executable from our source file
add_executable(example-app example-app.cpp)

# 4. Link the Torch libraries + TorchVision library
target_link_libraries(example-app 
    "${TORCH_LIBRARIES}"
    torchvision_library            # Name depends on how vision's CMakeLists defines it
)

set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
