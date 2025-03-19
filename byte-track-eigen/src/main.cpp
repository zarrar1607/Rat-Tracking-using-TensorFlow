#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Dense>

// ByteTrack headers (assumes you have these)
#include "BYTETracker.h"
#include "KalmanBBoxTrack.h"

// Simple function to convert BGR frame -> CHW float data in [0,1], normalized by ImageNet means/std
std::vector<float> preprocess(const cv::Mat& frame, const cv::Size& target_size = cv::Size(320, 320)) {
    // 1) Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    // 2) Resize
    cv::Mat resized;
    cv::resize(rgb, resized, target_size);

    // 3) Scale to [0,1] float
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    // 4) Normalize by ImageNet means/std
    cv::Mat channels[3];
    cv::split(resized, channels);
    float mean[3]    = {0.485f, 0.456f, 0.406f};
    float std_val[3] = {0.229f, 0.224f, 0.225f};
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std_val[i];
    }
    cv::merge(channels, 3, resized);

    // 5) Convert HWC -> CHW
    std::vector<cv::Mat> chw;
    cv::split(resized, chw);
    std::vector<float> input_tensor_values;
    // Concatenate channel planes
    for (int i = 0; i < 3; i++) {
        input_tensor_values.insert(input_tensor_values.end(),
            (float*)chw[i].datastart, (float*)chw[i].dataend);
    }
    return input_tensor_values;
}

int main() {
    // ----------------------------------------------------------------
    // 1) Initialize ONNX Runtime
    // ----------------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load your .onnx model (adjust the path accordingly).
    Ort::Session session(env, "../model.onnx", session_options);

    // For input name:
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    std::cout << "Model input name: " << input_name_alloc.get() << std::endl;

    // ----------------------------------------------------------------
    // 2) Video Capture
    // ----------------------------------------------------------------
    std::string video_path = "../Video/BaselineDark.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }

    // ----------------------------------------------------------------
    // 3) Setup ByteTracker
    // ----------------------------------------------------------------
    float track_thresh  = 0.2f;  // detection confidence threshold for tracking
    int track_buffer    = 30;    // how many frames to keep a lost track
    float match_thresh  = 0.8f;  // IoU matching threshold
    float fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0) fps = 30.f;
    BYTETracker byte_tracker(track_thresh, track_buffer, match_thresh, fps);

    // A container to store track histories: track_id -> list of points
    std::unordered_map<int, std::vector<cv::Point>> track_history;
    const size_t max_history = 30; // how many points to keep per track

    // ----------------------------------------------------------------
    // 4) Main Loop
    // ----------------------------------------------------------------
    cv::Size input_size(320, 320);
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            break; // end of video
        }

        int orig_w = frame.cols;
        int orig_h = frame.rows;

        // 4a) Preprocess frame for ONNX
        std::vector<float> input_tensor_values = preprocess(frame, input_size);
        std::array<int64_t, 4> input_shape = {1, 3, input_size.height, input_size.width};

        // Create input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()
        );

        // Names (assuming the model has 3 outputs: "boxes", "scores", "labels")
        const char* input_names[]  = { input_name_alloc.get() };
        const char* output_names[] = { "boxes", "scores", "labels" };

        // 4b) Inference
        auto start_time = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_names, &input_tensor, 1,
                                          output_names, 3);
        auto end_time = std::chrono::high_resolution_clock::now();
        double inf_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // 4c) Parse model outputs
        float* boxes_ptr  = output_tensors[0].GetTensorMutableData<float>(); // shape [N,4]
        float* scores_ptr = output_tensors[1].GetTensorMutableData<float>(); // shape [N]
        int64_t* labels_ptr = output_tensors[2].GetTensorMutableData<int64_t>(); // shape [N]
        // We'll only take the best detection above some threshold
        Ort::TensorTypeAndShapeInfo boxes_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> boxes_shape = boxes_info.GetShape();
        int64_t num_boxes = boxes_shape[0];

        float threshold = 0.2f;
        float best_score = threshold;
        int best_idx = -1;
        for (int i = 0; i < num_boxes; i++) {
            if (scores_ptr[i] > best_score) {
                best_score = scores_ptr[i];
                best_idx = i;
            }
        }

        // 4d) If we found a valid detection, create a single row [x, y, w, h, score]
        //     in the resized coordinate space
        if (best_idx >= 0) {
            float x1 = boxes_ptr[best_idx*4 + 0];
            float y1 = boxes_ptr[best_idx*4 + 1];
            float x2 = boxes_ptr[best_idx*4 + 2];
            float y2 = boxes_ptr[best_idx*4 + 3];
            float w = x2 - x1;
            float h = y2 - y1;

            Eigen::MatrixXf single_det(1, 5);
            single_det(0, 0) = x1;
            single_det(0, 1) = y1;
            single_det(0, 2) = w;
            single_det(0, 3) = h;
            single_det(0, 4) = best_score;

            // 4e) Update ByteTracker
            std::vector<KalmanBBoxTrack> results = byte_tracker.process_frame_detections(single_det);

            // We'll keep track of which track IDs are active in this frame
            std::unordered_set<int> active_ids;

            // 4f) Draw the tracks
            for (auto &trk : results) {
                Eigen::Vector4d box_tlbr = trk.tlbr(); // [x1, y1, x2, y2] in resized coords
                float bx1 = box_tlbr(0), by1 = box_tlbr(1);
                float bx2 = box_tlbr(2), by2 = box_tlbr(3);

                // Scale to original
                float scale_x = static_cast<float>(orig_w) / input_size.width;
                float scale_y = static_cast<float>(orig_h) / input_size.height;
                int orig_x1 = static_cast<int>(bx1 * scale_x);
                int orig_y1 = static_cast<int>(by1 * scale_y);
                int orig_x2 = static_cast<int>(bx2 * scale_x);
                int orig_y2 = static_cast<int>(by2 * scale_y);

                int track_id = trk.get_track_id();
                float track_score = trk.get_score();
                active_ids.insert(track_id);

                // Update track history with the center
                int cx = (orig_x1 + orig_x2)/2;
                int cy = (orig_y1 + orig_y2)/2;
                track_history[track_id].push_back(cv::Point(cx, cy));
                if (track_history[track_id].size() > max_history) {
                    track_history[track_id].erase(track_history[track_id].begin());
                }

                // Draw bounding box
                cv::rectangle(frame, cv::Point(orig_x1, orig_y1),
                              cv::Point(orig_x2, orig_y2),
                              cv::Scalar(0,255,0), 2);

                // Label
                std::string label = "ID:" + std::to_string(track_id) + " s:" + std::to_string(track_score);
                cv::putText(frame, label, cv::Point(orig_x1, std::max(0, orig_y1-5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);

                // Center
                cv::circle(frame, cv::Point(cx, cy), 3, cv::Scalar(0,0,255), -1);
            }

            // 4g) Remove track IDs from track_history that are no longer active
            for (auto it = track_history.begin(); it != track_history.end(); ) {
                if (active_ids.find(it->first) == active_ids.end()) {
                    // Not active => remove
                    it = track_history.erase(it);
                } else {
                    ++it;
                }
            }
        } else {
            // If no detection, we can choose to remove everything or keep it for a few frames
            // If you want to remove all lines whenever there's no detection:
            // track_history.clear();
        }

        // 4h) Draw polylines for the track histories
        for (const auto &kv : track_history) {
            const auto &pts = kv.second;
            if (pts.size() >= 2) {
                cv::polylines(frame, pts, false, cv::Scalar(230,230,230), 2);
            }
        }

        // Show inference time
        std::string inf_text = "Inference: " + std::to_string(inf_time) + " ms";
        cv::putText(frame, inf_text, cv::Point(10,30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,255), 2);

        cv::imshow("ByteTrack Demo", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
