#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <array>

// Preprocessing: convert frame (BGR) to normalized tensor data in CHW order.
std::vector<float> preprocess(const cv::Mat& frame, const cv::Size& target_size = cv::Size(320, 320)) {
    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    
    // Resize to target size
    cv::Mat resized;
    cv::resize(rgb, resized, target_size);
    
    // Convert to float and scale to [0,1]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    
    // Normalize using mean and std (same as training)
    cv::Mat channels[3];
    cv::split(resized, channels);
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std_val[3] = {0.229f, 0.224f, 0.225f};
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std_val[i];
    }
    cv::merge(channels, 3, resized);
    
    // Convert HWC to CHW
    std::vector<cv::Mat> chw;
    cv::split(resized, chw);
    std::vector<float> input_tensor_values;
    for (int i = 0; i < 3; i++) {
        input_tensor_values.insert(input_tensor_values.end(), 
            (float*)chw[i].datastart, (float*)chw[i].dataend);
    }
    return input_tensor_values;
}

int main() {
    // --------------------------
    // 1) ONNX Runtime Setup
    // --------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, "../model.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    std::cout << "Input name: " << input_name_alloc.get() << std::endl;
    
    // --------------------------
    // 2) Video Capture Setup
    // --------------------------
    std::string video_path = "../Video/BaselineDark.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }
    
    // --------------------------
    // 3) Main Loop
    // --------------------------
    cv::Mat frame;
    while (cap.read(frame)) {
        // Store original frame dimensions for later scaling
        int orig_w = frame.cols;
        int orig_h = frame.rows;
        cv::Size target_size(320, 320);
        
        // Preprocess the frame
        std::vector<float> input_tensor_values = preprocess(frame, target_size);
        
        // Create input tensor shape [1, 3, 320, 320]
        std::array<int64_t, 4> input_shape = {1, 3, target_size.height, target_size.width};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
        );
        
        // Set input and output names (assuming model outputs "boxes", "scores", "labels" in that order)
        const char* input_names[] = { input_name_alloc.get() };
        const char* output_names[] = { "boxes", "scores", "labels" };
        
        // Run inference and measure time
        auto start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 3);
        auto end = std::chrono::high_resolution_clock::now();
        double inf_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // --------------------------
        // 4) Process Model Outputs
        // --------------------------
        // Output 0: boxes (shape [N, 4])
        float* boxes_ptr = output_tensors[0].GetTensorMutableData<float>();
        Ort::TensorTypeAndShapeInfo boxes_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> boxes_shape = boxes_info.GetShape();
        int64_t num_boxes = boxes_shape[0];  // number of detections
        
        // Output 1: scores (shape [N])
        float* scores_ptr = output_tensors[1].GetTensorMutableData<float>();
        
        // Output 2: labels (shape [N]) - assuming int64_t type
        int64_t* labels_ptr = output_tensors[2].GetTensorMutableData<int64_t>();
        
        // Filter predictions: keep only detections with score > threshold and then select the best one.
        float threshold = 0.2f;
        int best_idx = -1;
        float best_score = threshold;
        for (int i = 0; i < num_boxes; i++) {
            if (scores_ptr[i] > best_score) {
                best_score = scores_ptr[i];
                best_idx = i;
            }
        }
        
        // --------------------------
        // 5) Scale and Draw Bounding Box
        // --------------------------
        if (best_idx != -1) {
            // Each box has 4 values: [x1, y1, x2, y2] in resized (320×320) coordinates.
            float x1 = boxes_ptr[best_idx * 4 + 0];
            float y1 = boxes_ptr[best_idx * 4 + 1];
            float x2 = boxes_ptr[best_idx * 4 + 2];
            float y2 = boxes_ptr[best_idx * 4 + 3];
            
            // Scale box back to original frame dimensions.
            float scale_x = static_cast<float>(orig_w) / static_cast<float>(target_size.width);
            float scale_y = static_cast<float>(orig_h) / static_cast<float>(target_size.height);
            int orig_x1 = static_cast<int>(x1 * scale_x);
            int orig_y1 = static_cast<int>(y1 * scale_y);
            int orig_x2 = static_cast<int>(x2 * scale_x);
            int orig_y2 = static_cast<int>(y2 * scale_y);
            
            // Draw the bounding box and label on the original frame.
            cv::rectangle(frame, cv::Point(orig_x1, orig_y1), cv::Point(orig_x2, orig_y2), cv::Scalar(0, 255, 0), 2);
            std::string label_text = "Rat: " + std::to_string(best_score);
            cv::putText(frame, label_text, cv::Point(orig_x1, orig_y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            // Draw the centroid as a red dot.
            int cx = (orig_x1 + orig_x2) / 2;
            int cy = (orig_y1 + orig_y2) / 2;
            cv::circle(frame, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);
        }
        
        // Display inference time on the frame.
        std::string inf_text = "Inference: " + std::to_string(inf_time) + " ms";
        cv::putText(frame, inf_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        
        cv::imshow("Predictions", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
