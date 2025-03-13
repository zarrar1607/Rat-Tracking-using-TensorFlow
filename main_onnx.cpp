#include <onnxruntime_cxx_api.h>

// Pseudocode for ONNX Runtime usage:
int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load the ONNX model
    Ort::Session session(env, "model.onnx", session_options);

    // Prepare input data
    std::vector<float> input_tensor_values(1 * 3 * 320 * 320, 1.0f); // Example data
    std::array<int64_t, 4> input_shape = {1, 3, 320, 320};

    // Create input tensor object from data values
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(),
        input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &"input",     // input node name
        &input_tensor,1,
        &"output",    // output node name
        1
    );

    // Process output_tensors[0] as needed
    return 0;
}
