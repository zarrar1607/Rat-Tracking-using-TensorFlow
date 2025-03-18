import numpy as np
from types import SimpleNamespace
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import your tracker classes
from trackers.byte_tracker import BYTETracker

# Helper: Convert a box from xyxy [x1, y1, x2, y2] to xywh [cx, cy, w, h]
def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy[:4]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.array([cx, cy, w, h], dtype=xyxy.dtype)

# Create a dummy results object with the attributes the tracker expects:
def create_dummy_results(dets):
    """
    Given dets (an array of shape (N, 5) with [x1, y1, x2, y2, score]),
    return a dummy object with attributes:
      - xywh: boxes in [cx, cy, w, h] format
      - conf: confidence scores
      - cls: class labels (here assumed to be 0 for all detections)
    """
    if len(dets) == 0:
        return SimpleNamespace(
            xywh=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
            cls=np.empty((0,), dtype=np.int32)
        )
    boxes_xywh = np.array([xyxy_to_xywh(det[:4]) for det in dets])
    conf = dets[:, 4]
    cls = np.zeros_like(conf, dtype=np.int32)  # assume single-class (0)
    return SimpleNamespace(xywh=boxes_xywh, conf=conf, cls=cls)

# Define dummy tracker arguments
class DummyArgs:
    track_buffer = 30
    track_high_thresh = 0.5   # detections with score >= 0.5 are high confidence
    track_low_thresh = 0.1    # detections with score > 0.1 are considered in low stage
    match_thresh = 0.8
    new_track_thresh = 0.6    # threshold for initiating a new track
    fuse_score = False

args = DummyArgs()

# Initialize the tracker (frame_rate in frames per second, adjust as needed)
tracker = BYTETracker(args=args, frame_rate=30)

# Simulated detection stream for 3 frames.
# Each detection is [x1, y1, x2, y2, score]
frames_detections = [
    np.array([[100, 100, 150, 150, 0.9],
              [300, 300, 350, 350, 0.85]], dtype=np.float32),
    np.array([[105, 105, 155, 155, 0.92],
              [305, 305, 355, 355, 0.87]], dtype=np.float32),
    np.array([[110, 110, 160, 160, 0.88]], dtype=np.float32)
]

# Dummy image info: assume original frame size of 480x640, and detector input size of 320x320.
img_info = (480, 640)
img_size = (320, 320)

# Process each simulated frame.
for frame_idx, dets in enumerate(frames_detections, start=1):
    print(f"--- Frame {frame_idx} ---")
    print("Raw detections (xyxy + score):", dets)
    
    # Create a dummy results object from the detections.
    results = create_dummy_results(dets)
    print(f'res: {results}')
    # Update tracker with the dummy results.
    tracked = tracker.update(results, img=None)
    # Print out the tracker output.
    # Each element of 'tracked' should be an array like:
    # [x1, y1, x2, y2, track_id, score, cls, idx]
    print("Tracked outputs:")
    print(tracked)
    print()

