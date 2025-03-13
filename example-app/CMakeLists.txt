cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

# 1. Find LibTorch
find_package(Torch REQUIRED)

# 2. Create the executable from our source file
add_executable(example-app example-app.cpp)

# 3. Link the Torch libraries to our executable
target_link_libraries(example-app "${TORCH_LIBRARIES}")

# 4. Set C++ standard to 17 (or 14, if you prefer)
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
