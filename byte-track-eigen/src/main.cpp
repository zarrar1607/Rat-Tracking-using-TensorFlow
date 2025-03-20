#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Dense>

// ByteTrack headers (adjust includes as needed)
#include "BYTETracker.h"
#include "KalmanBBoxTrack.h"

// ---------------------------------------------------------------------------
// 1) Preprocessing: convert BGR -> CHW float data, normalized by mean/std
// ---------------------------------------------------------------------------
std::vector<float> preprocess(const cv::Mat& frame, const cv::Size& target_size = cv::Size(320, 320)) {
    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    // Resize
    cv::Mat resized;
    cv::resize(rgb, resized, target_size);

    // Convert to float [0,1]
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    // Normalize by ImageNet means/std
    cv::Mat channels[3];
    cv::split(resized, channels);
    float mean[3]    = {0.485f, 0.456f, 0.406f};
    float std_val[3] = {0.229f, 0.224f, 0.225f};
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std_val[i];
    }
    cv::merge(channels, 3, resized);

    // Convert HWC -> CHW
    std::vector<cv::Mat> chw;
    cv::split(resized, chw); // 3 channels
    std::vector<float> input_tensor_values;
    for (int i = 0; i < 3; i++) {
        input_tensor_values.insert(input_tensor_values.end(),
            (float*)chw[i].datastart, (float*)chw[i].dataend);
    }
    return input_tensor_values;
}

// ---------------------------------------------------------------------------
// 2) Helper function: select top-k detections above threshold
//    Returns an (M,5) matrix in resized coords with columns [x, y, w, h, score].
// ---------------------------------------------------------------------------
Eigen::MatrixXf select_topk_detections(float* boxes_ptr, float* scores_ptr, int64_t num_boxes,
                                       float conf_thresh, int k /*top k*/,
                                       const cv::Size& input_size)
{
    // We'll gather all detections above conf_thresh, then pick top k by score
    // Each detection is [x1, y1, x2, y2], plus a separate 'score_ptr[i]'.
    // We'll store them in a vector so we can sort by descending score.
    struct Det {
        float x1, y1, x2, y2, score;
    };
    std::vector<Det> collected;
    collected.reserve(num_boxes);

    // 1) Collect all detections above threshold
    for (int i = 0; i < num_boxes; i++) {
        float score = scores_ptr[i];
        if (score >= conf_thresh) {
            float x1 = boxes_ptr[i*4 + 0];
            float y1 = boxes_ptr[i*4 + 1];
            float x2 = boxes_ptr[i*4 + 2];
            float y2 = boxes_ptr[i*4 + 3];
            collected.push_back({x1, y1, x2, y2, score});
        }
    }

    // 2) Sort by descending score
    std::sort(collected.begin(), collected.end(), [](const Det& a, const Det& b){
        return a.score > b.score;
    });

    // 3) Keep top k (or fewer if not enough detections)
    if ((int)collected.size() > k) {
        collected.resize(k);
    }

    // 4) Convert to an Eigen::MatrixXf [M,5]: [x, y, w, h, score]
    Eigen::MatrixXf output(collected.size(), 5);
    for (size_t i = 0; i < collected.size(); i++) {
        float w = collected[i].x2 - collected[i].x1;
        float h = collected[i].y2 - collected[i].y1;
        output(i,0) = collected[i].x1;
        output(i,1) = collected[i].y1;
        output(i,2) = w;
        output(i,3) = h;
        output(i,4) = collected[i].score;
    }

    return output;
}

int main()
{
    // ----------------------------------------------------------------
    // A) Initialize ONNX Runtime
    // ----------------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, "../model.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    std::cout << "Model input name: " << input_name_alloc.get() << std::endl;

    // ----------------------------------------------------------------
    // B) Video Capture
    // ----------------------------------------------------------------
    std::string video_path = "../Video/BaselineDark.mp4";
    // std::string video_path = "../Video/TestFile_video.mp4";
    // std::string video_path = "../Video/movie.mp4";
    // std::string video_path = "../Video/Cohort_1.mp4";
    // std::string video_path = "../Video/3_Mice.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }

    // ----------------------------------------------------------------
    // C) Setup ByteTracker
    // ----------------------------------------------------------------
    float track_thresh  = 0.2f;  // detection confidence threshold for tracking
    int track_buffer    = 30;    // keep lost tracks for 30 frames
    float match_thresh  = 0.8f;  // IoU matching threshold
    float fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0) fps = 30.f;
    BYTETracker byte_tracker(track_thresh, track_buffer, match_thresh, fps);

    // Container for track history
    std::unordered_map<int, std::vector<cv::Point>> track_history;
    const size_t max_history = 30; // store up to 30 points per track

    // We'll run inference on a 320×320 resized image
    cv::Size input_size(320, 320);

    // ----------------------------------------------------------------
    // D) Main Loop
    // ----------------------------------------------------------------
    cv::Mat frame;
    while (true) {
        // Attempt to read
        if (!cap.read(frame)) {
            // This means no more frames, or we can't decode the file
            std::cerr << "No more frames or cannot decode.\n";
            break;
        }
        if (frame.empty()) {
            std::cerr << "Got an empty frame. Possibly unsupported codec.\n";
            break;
        }
        int orig_w = frame.cols;
        int orig_h = frame.rows;

        // 1) Preprocess
        std::vector<float> input_tensor_values = preprocess(frame, input_size);

        // 2) Create input tensor
        std::array<int64_t, 4> input_shape = {1, 3, input_size.height, input_size.width};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()
        );

        // 3) Inference
        const char* input_names[]  = { input_name_alloc.get() };
        const char* output_names[] = { "boxes", "scores", "labels" };
        auto start_time = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 3
        );
        auto end_time = std::chrono::high_resolution_clock::now();
        double inf_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // 4) Parse outputs
        float* boxes_ptr  = output_tensors[0].GetTensorMutableData<float>(); // shape [N,4]
        float* scores_ptr = output_tensors[1].GetTensorMutableData<float>(); // shape [N]
        // int64_t* labels_ptr = output_tensors[2].GetTensorMutableData<int64_t>(); // shape [N], if needed
        auto boxes_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = boxes_info.GetShape();
        int64_t num_boxes = shape[0];

        // 5) Get top K=2 detections above threshold=0.2 in resized coords
        float conf_thresh = 0.2f;
        int top_k = 1;
        Eigen::MatrixXf topk_dets = select_topk_detections(
            boxes_ptr, scores_ptr, num_boxes, conf_thresh, top_k, input_size
        );

        // 6) If we have any detections, pass them to ByteTrack
        if (topk_dets.rows() > 0) {
            // The ByteTrack code expects [x, y, w, h, score] in the resized coordinate system
            std::vector<KalmanBBoxTrack> results = byte_tracker.process_frame_detections(topk_dets);

            // We'll track which IDs are active
            std::unordered_set<int> active_ids;

            // Draw each track
            for (auto &trk : results) {
                Eigen::Vector4d box_tlbr = trk.tlbr(); // [x1, y1, x2, y2] in resized coords
                float bx1 = box_tlbr(0);
                float by1 = box_tlbr(1);
                float bx2 = box_tlbr(2);
                float by2 = box_tlbr(3);

                // Scale back to original
                float scale_x = (float)orig_w / (float)input_size.width;
                float scale_y = (float)orig_h / (float)input_size.height;
                int orig_x1 = static_cast<int>(bx1 * scale_x);
                int orig_y1 = static_cast<int>(by1 * scale_y);
                int orig_x2 = static_cast<int>(bx2 * scale_x);
                int orig_y2 = static_cast<int>(by2 * scale_y);

                int track_id = trk.get_track_id();
                float track_score = trk.get_score();
                active_ids.insert(track_id);

                // Append center point to track history
                int cx = (orig_x1 + orig_x2)/2;
                int cy = (orig_y1 + orig_y2)/2;
                track_history[track_id].push_back(cv::Point(cx, cy));
                if (track_history[track_id].size() > max_history) {
                    track_history[track_id].erase(track_history[track_id].begin());
                }

                // Draw bounding box
                cv::rectangle(frame, cv::Point(orig_x1, orig_y1),
                              cv::Point(orig_x2, orig_y2),
                              cv::Scalar(0, 255, 0), 2);

                // Label
                std::string label = "ID:" + std::to_string(track_id) +
                                    " s:" + std::to_string(track_score);
                cv::putText(frame, label, cv::Point(orig_x1, std::max(0, orig_y1 - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

                // Center
                cv::circle(frame, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);
            }

            // Remove any old track IDs not active
            for (auto it = track_history.begin(); it != track_history.end();) {
                if (active_ids.find(it->first) == active_ids.end()) {
                    it = track_history.erase(it);
                } else {
                    ++it;
                }
            }
        }
        else {
            // If no detections, optionally clear track history:
            // track_history.clear();
        }

        // 7) Draw polylines for each track's history
        for (const auto& kv : track_history) {
            const auto &pts = kv.second;
            if (pts.size() >= 2) {
                cv::polylines(frame, pts, false, cv::Scalar(230, 0, 0), 2);
            }
        }

        // Show inference time
        std::string inf_text = "Inference: " + std::to_string(inf_time) + " ms";
        cv::putText(frame, inf_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,255), 2);

        cv::imshow("ByteTrack with Top-K=2", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
