import cv2
import os

# Define classes (you can modify this list based on your dataset)
CLASSES = ['rat']  # Add more classes as needed
current_class_id = 0  # Default class ID

# Global variables for drawing and annotation
drawing = False
ix, iy, fx, fy = -1, -1, -1, -1
img = None
annotations = []
cap = None
frame_count = 0
original_frames = {}

# Mouse callback function to draw bounding boxes
def draw_box(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, img, annotations

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Frame', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 2)
        cv2.imshow('Frame', img)

        # Save the annotation (normalized coordinates)
        h, w = img.shape[:2]
        x_center = ((ix + fx) / 2) / w
        y_center = ((iy + fy) / 2) / h
        width = abs(fx - ix) / w
        height = abs(fy - iy) / h

        annotations.append(f"{current_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

# Function to change the current class
def change_class(key):
    global current_class_id
    if key == ord('n'):  # Next class
        current_class_id = (current_class_id + 1) % len(CLASSES)
    elif key == ord('p'):  # Previous class
        current_class_id = (current_class_id - 1) % len(CLASSES)
    print(f"Current class: {CLASSES[current_class_id]}")

# Trackbar callback function
def on_trackbar_change(frame_number):
    global cap, img, annotations, frame_count, original_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set the video to the selected frame
    ret, original_frame = cap.read()
    if ret:
        img = original_frame.copy()  # Make a copy for drawing bounding boxes
        annotations = []  # Clear annotations for the new frame
        frame_count = frame_number  # Update frame_count to match the trackbar position
        original_frames[frame_count] = original_frame  # Store the original frame
        cv2.imshow('Frame', img)

# Main function to process video and create annotations
def annotate_video(video_title, video_path, output_dir):
    global img, annotations, drawing, ix, iy, fx, fy, cap, frame_count, original_frames

    # Create output directories if they don't exist
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
    frame_count = 0

    # Dictionary to store original frames for each frame number
    original_frames = {}

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', draw_box)

    # Create a trackbar for scrolling through frames
    cv2.createTrackbar('Frame', 'Frame', 0, total_frames - 1, on_trackbar_change)

    while True:
        ret, original_frame = cap.read()  # Read the original frame
        if not ret:
            break

        img = original_frame.copy()  # Make a copy for drawing bounding boxes
        annotations = []
        drawing = False
        ix, iy, fx, fy = -1, -1, -1, -1

        # Store the original frame for the current frame_count
        original_frames[frame_count] = original_frame

        # Update the trackbar position
        cv2.setTrackbarPos('Frame', 'Frame', frame_count)

        cv2.imshow('Frame', img)
        print(f"Frame {frame_count}: Press 's' to save annotations, 'n'/'p' to change class, 'c' to clear boxes, 'q' to quit.")

        while True:
            key = cv2.waitKey(30) & 0xFF  # Adjust delay as needed
            if key == ord('s'):  # Save annotations
                if annotations:
                    # Retrieve the original frame for the current frame_count
                    original_frame = original_frames.get(frame_count, None)
                    if original_frame is not None:
                        # Save the original image without bounding box
                        image_file = os.path.join(images_dir, f"{video_title}_frame_{frame_count:06d}.jpg")
                        cv2.imwrite(image_file, original_frame)
                        
                        # Save annotations
                        annotation_file = os.path.join(labels_dir, f"{video_title}_frame_{frame_count:06d}.txt")
                        with open(annotation_file, 'w') as f:
                            f.write("\n".join(annotations))
                        print(f"Annotations and image saved for frame {frame_count}")
                break

            elif key == ord('n') or key == ord('p'):  # Change class
                change_class(key)

            elif key == ord('c'):  # Clear bounding boxes
                # Reset image to original and clear annotations
                img = original_frames[frame_count].copy()
                annotations = []
                cv2.imshow('Frame', img)
                print("Cleared bounding boxes for current frame.")

            elif key == ord('q'):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                return

        # Only increment frame_count if the user hasn't moved the trackbar
        if cv2.getTrackbarPos('Frame', 'Frame') == frame_count:
            frame_count += 1
        else:
            # If the trackbar was moved, update frame_count to match the trackbar position
            frame_count = cv2.getTrackbarPos('Frame', 'Frame')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_title = "BaselineDark"  # Use a title without extension if desired
    video_path = ".\\Video\\" + video_title + ".mp4"  # Replace with your video path
    output_dir = "annotations"  # Directory to save annotations and images
    annotate_video(video_title, video_path, output_dir)
