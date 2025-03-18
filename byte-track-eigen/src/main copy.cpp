#include "BYTETracker.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Helper function to convert an Eigen::Vector4d (assumed to be in [x1, y1, x2, y2] format)
// to a cv::Rect2f.
cv::Rect2f eigenToRect2f(const Eigen::Vector4d &box) {
    float x = static_cast<float>(box[0]);
    float y = static_cast<float>(box[1]);
    float width = static_cast<float>(box[2] - box[0]);
    float height = static_cast<float>(box[3] - box[1]);
    return cv::Rect2f(x, y, width, height);
}

int main() {
    // Tracker parameters.
    float track_thresh = 0.23f;
    int track_buffer = 30;
    float match_thresh = 0.8f;
    int frame_rate = 30;

    // Create the BYTETracker instance.
    BYTETracker tracker(track_thresh, track_buffer, match_thresh, frame_rate);

    // Create an Eigen matrix for detections.
    // Each row is: [x1, y1, x2, y2, score]
    Eigen::MatrixXf detections(2, 5);
    detections << 100, 100, 150, 150, 0.9,
                  300, 300, 350, 350, 0.85;

    // Process detections for frame 1.
    std::vector<KalmanBBoxTrack> tracks = tracker.process_frame_detections(detections);

    // Print track results.
    for (const auto &track : tracks) {
        // Get the bounding box in [x1, y1, x2, y2] format using tlbr().
        Eigen::Vector4d bbox = track.tlbr();
        cv::Rect2f box = eigenToRect2f(bbox);
        
        // Get the track ID and score using public accessor methods.
        int id = track.get_track_id();    // Make sure this is public in your class.
        float score = track.get_score();    // Make sure this is public in your class.

        std::cout << "Track ID: " << id
                  << ", Box: [" << box.x << ", " << box.y << ", "
                  << box.width << ", " << box.height << "], Score: "
                  << score << std::endl;
    }

    return 0;
}
