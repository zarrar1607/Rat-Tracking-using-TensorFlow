#include <torch/script.h>  // One-stop header.
#include <iostream>
#include <memory>

int main() {
    // Load the TorchScript model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("model.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }
    std::cout << "Model loaded successfully.\n";

    // Create a dummy input tensor. Adjust the dimensions as required (e.g., batch size, channels, height, width).
    torch::Tensor input = torch::rand({1, 3, 320, 320});
    
    // Run inference. Note that detection models may return complex data structures (e.g., dictionaries with boxes, labels, scores)
    auto output = module.forward({input});

    // Print the output (for simple inspection)
    std::cout << output << "\n";

    return 0;
}
