#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <array>

int main() {
    // Create environment and session options
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load the ONNX model
    Ort::Session session(env, "model.onnx", session_options);

    // Prepare input data
    std::vector<float> input_tensor_values(1 * 3 * 320 * 320, 1.0f);
    std::array<int64_t, 4> input_shape = {1, 3, 320, 320};

    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_tensor_values.data(), 
        input_tensor_values.size(),
        input_shape.data(), 
        input_shape.size()
    );

    // Define arrays of const char* for node names
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,        // array of input name(s)
        &input_tensor, 1,   // pointer to input tensor, number of inputs
        output_names, 1     // array of output name(s), number of outputs
    );

    // Example: print shape of first output
    auto& output_tensor = output_tensors.front();
    auto type_and_shape = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = type_and_shape.GetShape();
    std::cout << "Output shape: ";
    for (auto dim : output_shape) {
        std::cout << dim << " ";
    }
    std::cout << "\nInference done.\n";

    return 0;
}
