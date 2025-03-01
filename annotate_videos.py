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

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --------------------------
# Global variables for mouse drawing
# --------------------------
drawing = False         # True if mouse is pressed
ix, iy = -1, -1         # Initial coordinates on mouse down
current_bbox = None     # (x1, y1, x2, y2)
frame_orig = None       # The original frame (saved to disk)
frame_display = None    # A copy used only for drawing/visualization

def draw_rectangle(event, x, y, flags, param):
    """
    Mouse callback function to draw a rectangle (bounding box).
    We only draw on frame_display so the saved frame remains clean.
    """
    global ix, iy, drawing, current_bbox, frame_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Redraw a fresh copy so the rectangle isn't duplicated
            temp_display = frame_display.copy()
            cv2.rectangle(temp_display, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Video Annotation', temp_display)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Store the final bounding box
        current_bbox = (ix, iy, x, y)
        # Draw the rectangle permanently on frame_display
        cv2.rectangle(frame_display, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('Video Annotation', frame_display)

# --------------------------
# Open the video file
# --------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

cv2.namedWindow('Video Annotation', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video Annotation', draw_rectangle)

annotations_list = []
frame_number = 0

print("Instructions:")
print("  - Draw a bounding box on the frame using the mouse (click and drag).")
print("  - Press 's' to save the annotation (original frame + bbox coords).")
print("  - Press 'n' to move to the next frame without saving.")
print("  - Press 'q' to quit.")

# --------------------------
# Process each frame
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video or encountered a read error.")
        break

    frame_number += 1
    # Keep an unmodified copy for saving
    frame_orig = frame.copy()
    # This copy is for drawing/visualizing
    frame_display = frame.copy()
    current_bbox = None

    # Show the current (clean) frame
    cv2.imshow('Video Annotation', frame_display)
    print(f"Frame {frame_number}: Draw a bounding box if needed, then press 's' or 'n' or 'q'.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Exiting annotation.")
            cap.release()
            cv2.destroyAllWindows()
            # After quitting, save any existing annotations below
            goto_save = False
            break
        
        elif key == ord('s'):
            # Save the frame and annotation if a bounding box was drawn
            if current_bbox is not None:
                # Save the original frame (no bounding box drawn)
                filename = f"{VIDEO_FILENAME}_frame_{frame_number:05d}.jpg"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, frame_orig)
                
                # Store bounding box as a string "x1,y1,x2,y2"
                (x1, y1, x2, y2) = current_bbox
                bbox_str = f"{x1},{y1},{x2},{y2}"
                
                annotations_list.append({
                    'frame': frame_number,
                    'filename': filename,
                    'bbox': bbox_str
                })
                print(f"Saved annotation for frame {frame_number} with bbox {bbox_str}.")
            else:
                print("No bounding box drawn. Skipping annotation for this frame.")
            
            goto_save = True
            break
        
        elif key == ord('n'):
            print(f"Skipping frame {frame_number} without saving.")
            goto_save = True
            break
        
        # Keep looping until 's', 'n', or 'q' is pressed
        # This allows time to draw the bounding box with the mouse
    
    if not ret or not goto_save:
        # If we reached the end or user pressed 'q', stop the outer loop
        break

cap.release()
cv2.destroyAllWindows()

# --------------------------
# Save annotations to CSV
# --------------------------
if annotations_list:
    df_annotations = pd.DataFrame(annotations_list)
    csv_path = os.path.join(OUTPUT_DIR, ANNOTATIONS_CSV)
    df_annotations.to_csv(csv_path, index=False)
    print(f"Annotations saved to {csv_path}.")
else:
    print("No annotations were saved.")
