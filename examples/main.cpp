#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <string>

// Helper: Preprocess a cv::Mat frame into a float array for model input
// This example resizes to 320x320, converts BGR->RGB, normalizes to [0..1].
static std::vector<float> preprocessFrame(const cv::Mat& frame, int input_w, int input_h) {
    // 1) Resize
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_w, input_h));

    // 2) Convert BGR -> RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 3) Normalize to [0..1] or any other normalization your model expects
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    // 4) HWC -> CHW
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
    // 1. Load ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 2. Load the ONNX model
    //    Adjust path if needed
    std::string model_path = "model.onnx";
    Ort::Session session(env, model_path.c_str(), session_options);

    // 3. OpenCV: Open a video file
    std::string video_path = "../Video/TestFile_video.mp4"; // Update to your video file path
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }

    // Suppose your model expects 320x320
    const int input_w = 320;
    const int input_h = 320;

    // Prepare input node info
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // Inference loop
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            break; // end of video
        }

        // Preprocess
        std::vector<float> input_data = preprocessFrame(frame, input_w, input_h);

        // Create input tensor
        std::array<int64_t, 4> input_shape = {1, 3, input_h, input_w};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );

        // Assume the model outputs Nx6: [x1, y1, x2, y2, score, class_id]
        auto& output_tensor = output_tensors.front();
        auto type_and_shape = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape = type_and_shape.GetShape();
        // out_shape might be [N, 6]

        // Access the raw data
        float* out_data = output_tensor.GetTensorMutableData<float>();
        int64_t num_boxes = out_shape[0];
        int64_t elements_per_box = out_shape[1];

        // Draw boxes
        for (int64_t i = 0; i < num_boxes; ++i) {
            float x1 = out_data[i * elements_per_box + 0];
            float y1 = out_data[i * elements_per_box + 1];
            float x2 = out_data[i * elements_per_box + 2];
            float y2 = out_data[i * elements_per_box + 3];
            float score = out_data[i * elements_per_box + 4];
            float cls   = out_data[i * elements_per_box + 5];

            // Optionally filter out low scores
            if (score < 0.5f) continue;

            // The coordinates x1,y1,x2,y2 are presumably in the original image scale
            // If your model returns them in [0..320], you might need to scale them up
            // to the original frame size
            float scale_x = static_cast<float>(frame.cols) / input_w;
            float scale_y = static_cast<float>(frame.rows) / input_h;
            int rx1 = static_cast<int>(x1 * scale_x);
            int ry1 = static_cast<int>(y1 * scale_y);
            int rx2 = static_cast<int>(x2 * scale_x);
            int ry2 = static_cast<int>(y2 * scale_y);

            cv::rectangle(frame, cv::Point(rx1, ry1), cv::Point(rx2, ry2), cv::Scalar(0, 255, 0), 2);
            std::string label_text = "Score: " + std::to_string(score) + " cls: " + std::to_string(cls);
            cv::putText(frame, label_text, cv::Point(rx1, ry1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        }

        // Show result
        cv::imshow("Detections", frame);
        if (cv::waitKey(1) == 27) { // press 'ESC' to quit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
