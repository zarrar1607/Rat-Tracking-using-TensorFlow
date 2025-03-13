#include <torch/script.h> // One-stop header for TorchScript
#include <iostream>
#include <memory>

int main() {
    // 1. Load the TorchScript model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("model.pt");
        // If your model requires CUDA, you can do: module.to(torch::kCUDA);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model.pt file: " << e.what() << "\n";
        return -1;
    }

    std::cout << "Model loaded successfully.\n";

    // 2. Create an example input tensor
    //    Adjust shape to match your model (e.g., (1,3,320,320) for images).
    at::Tensor input = torch::rand({1, 3, 320, 320});

    // 3. Run forward pass
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // For object detection models, the output might be a list of dicts, etc.
    // But here's a minimal example that expects a single Tensor output:
    at::Tensor output = module.forward(inputs).toTensor();

    // 4. Print the output shape
    std::cout << "Output shape: " << output.sizes() << "\n";

    return 0;
}
