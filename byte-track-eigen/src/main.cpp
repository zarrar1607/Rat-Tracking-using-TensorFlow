#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <array>
#include <Eigen/Dense>

// Include ByteTrack headers
#include "BYTETracker.h"           // Provides class BYTETracker
#include "KalmanBBoxTrack.h"       // Provides class KalmanBBoxTrack

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
    // std::string video_path = "../Video/TestFile_video.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }
    
    // --------------------------
    // 2) Initialize ByteTrack Tracker
    // --------------------------
    float track_thresh = 0.2f;   // detection confidence threshold for tracking
    int track_buffer = 100;       // track buffer size (number of frames to keep lost tracks)
    float match_thresh = 0.8f;   // IoU matching threshold
    // Use the video's FPS or a default value
    float fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0) fps = 30.0f;
    BYTETracker byte_tracker(track_thresh, track_buffer, match_thresh, fps);


    // --------------------------
    // 4) Main Loop
    // --------------------------
    cv::Mat frame;
    while (cap.read(frame)) {
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
        // 5) Process Model Outputs
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
        // 7) Scale and Draw Bounding Box
        // --------------------------
        if (best_idx != -1) {
            // Each box has 4 values: [x1, y1, x2, y2] in resized (320×320) coordinates.
            float x1 = boxes_ptr[best_idx * 4 + 0];
            float y1 = boxes_ptr[best_idx * 4 + 1];
            float x2 = boxes_ptr[best_idx * 4 + 2];
            float y2 = boxes_ptr[best_idx * 4 + 3];

            float w = x2 - x1;
            float h = y2 - y1;

            // Create a single detection matrix [1×5]: [orig_x1, orig_y1, orig_x2, orig_y2, best_score]
            Eigen::MatrixXf single_det(1, 5);
            single_det(0, 0) = static_cast<float>(x1);
            single_det(0, 1) = static_cast<float>(y1);
            single_det(0, 2) = static_cast<float>(w);
            single_det(0, 3) = static_cast<float>(h);
            single_det(0, 4) = best_score;

            // Feed the detection to ByteTrack
            std::vector<KalmanBBoxTrack> tracks = byte_tracker.process_frame_detections(single_det);

            // Draw the returned tracks on the frame
            for (auto &trk : tracks) {
                // Assume your KalmanBBoxTrack has a method tlbr() that returns a Eigen::Vector4d [x1, y1, x2, y2]
                Eigen::Vector4d box_tlbr = trk.tlbr();
                float bx1 = box_tlbr(0);
                float by1 = box_tlbr(1);
                float bx2 = box_tlbr(2);
                float by2 = box_tlbr(3);

                // Scale box back to original frame dimensions.
                float scale_x = static_cast<float>(orig_w) / static_cast<float>(target_size.width);
                float scale_y = static_cast<float>(orig_h) / static_cast<float>(target_size.height);
                int orig_x1 = static_cast<int>(bx1 * scale_x);
                int orig_y1 = static_cast<int>(by1 * scale_y);
                int orig_x2 = static_cast<int>(bx2 * scale_x);
                int orig_y2 = static_cast<int>(by2 * scale_y);

                int track_id = trk.get_track_id();
                float track_score = trk.get_score();

                // Draw bounding box
                cv::rectangle(frame, cv::Point((int)orig_x1, (int)orig_y1),
                            cv::Point((int)orig_x2, (int)orig_y2),
                            cv::Scalar(0, 255, 0), 2);

                // Draw label with track ID and score
                std::string lbl = "ID:" + std::to_string(track_id) + " s:" + std::to_string(track_score);
                cv::putText(frame, lbl, cv::Point((int)orig_x1, std::max(0, (int)(orig_y1 - 5))),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

                // Optionally, draw the centroid as a red dot
                int cx = (int)((orig_x1 + orig_x2) / 2.0f);
                int cy = (int)((orig_y1 + orig_y2) / 2.0f);
                cv::circle(frame, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Display inference time on the frame.
        std::string inf_text = "Inference: " + std::to_string(inf_time) + " ms";
        cv::putText(frame, inf_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        
        cv::imshow("ByteTrack", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
