import cv2
import pandas as pd
import os

# --------------------------
# Configuration
# --------------------------
VIDEO_FILENAME = 'TestFile_video'
VIDEO_PATH = './Video/' + VIDEO_FILENAME + '.mp4'
OUTPUT_DIR = './annotated_frames'
ANNOTATIONS_CSV = 'annotations.csv'
csv_path = os.path.join(OUTPUT_DIR, ANNOTATIONS_CSV)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --------------------------
# Open the video and get total frames
# --------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

# --------------------------
# Data structures
# --------------------------
# We store exactly one bounding box per frame in a list.
# If a frame is not annotated, its entry remains None.
annotations = [None] * total_frames

# Global "current frame index"
current_index = 0

# Global for the current bounding box being drawn
drawing = False
ix, iy = -1, -1

# For display, we have two images:
#   frame_orig: the original from the video
#   frame_display: a copy for drawing
frame_orig = None
frame_display = None

# --------------------------
# Mouse Callback
# --------------------------
def draw_rectangle(event, x, y, flags, param):
    """
    Allows the user to click and drag a rectangle on frame_display.
    When the mouse is released, we store that rectangle in annotations[current_index].
    """
    global ix, iy, drawing, frame_display, frame_orig, annotations, current_index
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw on a temporary copy so we don't accumulate rectangles
            temp_display = frame_orig.copy()
            # If there's already a stored box for this frame, draw it first
            if annotations[current_index] is not None:
                (sx1, sy1, sx2, sy2) = annotations[current_index]
                cv2.rectangle(temp_display, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            # Draw the in-progress rectangle
            cv2.rectangle(temp_display, (ix, iy), (x, y), (0, 255, 0), 2)
            frame_display = temp_display
            cv2.imshow('Video Annotation', frame_display)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Final bounding box
        x1, y1, x2, y2 = ix, iy, x, y
        # Normalize coordinates if user drags in reverse
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        annotations[current_index] = (x1, y1, x2, y2)
        # Update the display to show the final box
        frame_display = frame_orig.copy()
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Video Annotation', frame_display)

# --------------------------
# Helper: Load and display a given frame index
# --------------------------
def load_frame(frame_idx):
    """
    Seek to frame_idx, read it, update global frame_orig & frame_display,
    and draw any existing bounding box on it.
    """
    global cap, frame_orig, frame_display
    # Seek to that frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_idx}.")
        return
    frame_orig = frame.copy()
    frame_display = frame.copy()
    # If there's a stored bounding box for this frame, draw it
    if annotations[frame_idx] is not None:
        (x1, y1, x2, y2) = annotations[frame_idx]
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Video Annotation', frame_display)

# --------------------------
# Trackbar callback
# --------------------------
def on_trackbar(pos):
    """
    Called whenever the trackbar changes.
    We'll set current_index and load that frame.
    """
    global current_index
    current_index = pos
    load_frame(current_index)

# --------------------------
# Setup Window, Trackbar, Mouse
# --------------------------
cv2.namedWindow('Video Annotation', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video Annotation', draw_rectangle)
# Create a trackbar named 'Frame' that goes from 0 to total_frames-1
cv2.createTrackbar('Frame', 'Video Annotation', 0, total_frames - 1, on_trackbar)

print("Instructions:")
print("  - Use the scroll bar (trackbar) to jump to any frame.")
print("  - Or press 'a'/'d' to move backward/forward by 1 frame.")
print("  - Draw a bounding box on the frame using the mouse (click and drag).")
print("  - Press 's' to confirm/save the bounding box for the current frame, then move forward.")
print("  - Press 'n' to discard the bounding box on the current frame, then move forward.")
print("  - Press 'q' or ESC to quit. (ESC also triggers final saving.)")

# Initially load the first frame
load_frame(0)

# --------------------------
# Main Loop
# --------------------------
while True:
    key = cv2.waitKey(50) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or 'q' => finalize & exit
        print("Exiting annotation.")
        break
    elif key == ord('a'):  # previous frame
        pos = max(current_index - 1, 0)
        cv2.setTrackbarPos('Frame', 'Video Annotation', pos)
    elif key == ord('d'):  # next frame
        pos = min(current_index + 1, total_frames - 1)
        cv2.setTrackbarPos('Frame', 'Video Annotation', pos)
    elif key == ord('s'):
        # 's' => keep the bounding box, then move forward
        if annotations[current_index] is not None:
            print(f"Saved annotation for frame {current_index} with bbox {annotations[current_index]}.")
        else:
            print("No bounding box drawn. Skipping annotation for this frame.")
        next_idx = min(current_index + 1, total_frames - 1)
        cv2.setTrackbarPos('Frame', 'Video Annotation', next_idx)
    elif key == ord('n'):
        # 'n' => discard bounding box, then move forward
        if annotations[current_index] is not None:
            print(f"Discarding annotation for frame {current_index}.")
            annotations[current_index] = None
        next_idx = min(current_index + 1, total_frames - 1)
        cv2.setTrackbarPos('Frame', 'Video Annotation', next_idx)

# Cleanup the annotation window and release video capture
cv2.destroyAllWindows()
cap.release()

# --------------------------
# Save annotated frames & CSV
# --------------------------
csv_rows = []
cap = cv2.VideoCapture(VIDEO_PATH)  # Re-open to read frames from start
for i, bbox in enumerate(annotations):
    if bbox is not None:
        # Seek to frame i again and read the original frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        filename = f"{VIDEO_FILENAME}_frame_{i:05d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        # Save the clean frame (no box drawn)
        cv2.imwrite(filepath, frame)
        (x1, y1, x2, y2) = bbox
        bbox_str = f"{x1},{y1},{x2},{y2}"
        csv_rows.append({
            'frame_index': i,
            'filename': filename,
            'bbox': bbox_str
        })
cap.release()

if csv_rows:
    df_new = pd.DataFrame(csv_rows)
    # If the CSV already exists, load it and update (replace rows for same frame_index)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # Combine new and existing annotations; keep the last entry per frame_index.
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=['frame_index'], keep='last')
        df_combined.to_csv(csv_path, index=False)
        print(f"Updated annotations saved to {csv_path}.")
    else:
        df_new.to_csv(csv_path, index=False)
        print(f"Saved {len(csv_rows)} annotations to {csv_path}.")
else:
    print("No annotations were saved.")
