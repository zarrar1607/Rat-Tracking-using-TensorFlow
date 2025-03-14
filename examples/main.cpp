#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <string>

// Helper function: Preprocess a frame into [1, 3, 320, 320] float data
static std::vector<float> preprocessFrame(const cv::Mat& frame, int input_w, int input_h) {
    // 1) Resize to (320, 320)
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_w, input_h));

    // 2) Convert BGR -> RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 3) Convert to float32 in [0..1]
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    // 4) Reorder from HWC to CHW
    std::vector<float> input_tensor_values(input_w * input_h * 3);
    int channel_size = input_w * input_h;
    for (int y = 0; y < input_h; ++y) {
        for (int x = 0; x < input_w; ++x) {
            cv::Vec3f pixel = resized.at<cv::Vec3f>(y, x);
            // pixel = [R, G, B]
            input_tensor_values[0 * channel_size + y * input_w + x] = pixel[0];
            input_tensor_values[1 * channel_size + y * input_w + x] = pixel[1];
            input_tensor_values[2 * channel_size + y * input_w + x] = pixel[2];
        }
    }
    return input_tensor_values;
}

int main(int argc, char** argv) {
    // --------------------------
    // 1) ONNX Runtime Setup
    // --------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load the ONNX model (adjust the path if needed)
    std::string model_path = "model.onnx"; 
    Ort::Session session(env, model_path.c_str(), session_options);

    // --------------------------
    // 2) OpenCV: Open a video
    // --------------------------
    // Adjust this to your actual video file path
    std::string video_path = "../Video/BaselineDark.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }

    // The model expects 320x320
    const int input_w = 320;
    const int input_h = 320;

    // Confidence threshold (matching your Python logic)
    float conf_threshold = 0.2f;

    // The ONNX model has 1 input node: "input"
    const char* input_names[] = {"input"};

    // The ONNX model has 3 output nodes: "boxes", "scores", "labels"
    // as you saw in Netron
    const char* output_names[] = {"boxes", "scores", "labels"};
    size_t num_outputs = 3;

    // --------------------------
    // 3) Video Inference Loop
    // --------------------------
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            break;  // end of video
        }

        // 3.1) Preprocess the frame to [1,3,320,320]
        std::vector<float> input_data = preprocessFrame(frame, input_w, input_h);
        std::array<int64_t, 4> input_shape = {1, 3, input_h, input_w};

        // Create ONNX Runtime input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        // 3.2) Run inference requesting the 3 outputs
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, num_outputs
        );
        // output_tensors[0] => bounding boxes Nx4
        // output_tensors[1] => scores Nx
        // output_tensors[2] => labels Nx

        // --------------------------
        // 4) Parse the 3 outputs
        // --------------------------

        // 4.1) bounding boxes
        auto& boxes_tensor = output_tensors[0];
        auto boxes_info = boxes_tensor.GetTensorTypeAndShapeInfo();
        float* boxes_data = boxes_tensor.GetTensorMutableData<float>();
        std::vector<int64_t> boxes_shape = boxes_info.GetShape(); // e.g. [N,4]
        int64_t num_boxes = boxes_shape[0];  // how many

        // 4.2) scores
        auto& scores_tensor = output_tensors[1];
        auto scores_info = scores_tensor.GetTensorTypeAndShapeInfo();
        float* scores_data = scores_tensor.GetTensorMutableData<float>();
        // shape might be [N]

        // 4.3) labels
        auto& labels_tensor = output_tensors[2];
        auto labels_info = labels_tensor.GetTensorTypeAndShapeInfo();
        float* labels_data = labels_tensor.GetTensorMutableData<float>();
        // shape might be [N]

        // --------------------------
        // 5) Keep Single Highest
        // --------------------------
        float best_score = -1.0f;
        int best_idx = -1;
        for (int i = 0; i < num_boxes; ++i) {
            float score = scores_data[i];
            if (score >= conf_threshold && score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        // If we found a detection
        if (best_idx >= 0) {
            float x1 = boxes_data[best_idx*4 + 0];
            float y1 = boxes_data[best_idx*4 + 1];
            float x2 = boxes_data[best_idx*4 + 2];
            float y2 = boxes_data[best_idx*4 + 3];
            float cls = labels_data[best_idx];

            // If coords are in [0..320], scale them to the original frame
            float scale_x = static_cast<float>(frame.cols) / input_w;
            float scale_y = static_cast<float>(frame.rows) / input_h;
            int rx1 = static_cast<int>(x1 * scale_x);
            int ry1 = static_cast<int>(y1 * scale_y);
            int rx2 = static_cast<int>(x2 * scale_x);
            int ry2 = static_cast<int>(y2 * scale_y);

            // Draw bounding box and label
            cv::rectangle(frame, cv::Point(rx1, ry1), cv::Point(rx2, ry2),
                          cv::Scalar(0, 255, 0), 2);
            std::string label_text = "Rat: " + std::to_string(best_score);
            cv::putText(frame, label_text, cv::Point(rx1, ry1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        }

        // 6) Show the result
        cv::imshow("Predictions", frame);
        if (cv::waitKey(1) == 27) { // ESC key
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
