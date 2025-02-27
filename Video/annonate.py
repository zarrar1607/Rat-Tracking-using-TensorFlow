import os
import cv2
import json
import csv

# Global directories
EXTRACTED_FRAMES_DIR = "dataset"
LOCAL_VIDEO_DIR = "C:/Machine Learning/Dataset"
os.makedirs(EXTRACTED_FRAMES_DIR, exist_ok=True)

# CSV output file
OUTPUT_CSV = "annotations.csv"

def fix_video_filename(video_path):
    """
    Given a path like '/data/upload/1/0492ca76-Baseline.mp4',
    returns 'Baseline.mp4'.
    """
    filename = os.path.basename(video_path)
    if '-' in filename:
        return filename.split('-', 1)[1]
    
    local_video_path = os.path.join(LOCAL_VIDEO_DIR, filename)
    return local_video_path

def extract_frames_and_save(json_path="../Annotation_and_Data_creations/rat_data.json"):
    """
    Reads the JSON annotation file, extracts frames from the local videos,
    saves the images to EXTRACTED_FRAMES_DIR, and writes a CSV file containing:
      - annotation_id: The id from the bounding box annotation.
      - video: The fixed local video filename.
      - frame: The frame number from which the image was extracted.
      - bounding_box: A string of pixel coordinates "x_min, y_min, x_max, y_max".
      - image_path: Path to the saved frame image.
    """
    # Open the CSV file for writing
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["annotation_id", "video", "frame", "bounding_box", "image_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Load the JSON file
        with open(json_path, "r") as f:
            data = json.load(f)  # data is a list of tasks

        # Iterate over tasks
        for task in data:
            video_path = task["data"]["video"]
            fixed_video_filename = fix_video_filename(video_path)
            local_video_path = os.path.join(LOCAL_VIDEO_DIR, fixed_video_filename)
            print(f"Processing video: {fixed_video_filename}")

            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                print(f"Error opening video: {fixed_video_filename}")
                continue

            # Get video dimensions (for converting percentages to pixels)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Iterate over annotations
            for annotation in task["annotations"]:
                # Use the bbox id as the annotation id
                annotation_id = annotation["result"][0]["id"] if annotation["result"] else "unknown"
                for bbox in annotation["result"]:
                    if bbox.get("type") != "videorectangle":
                        continue

                    sequence = bbox["value"]["sequence"]
                    # Process each frame in the sequence
                    for frame_data in sequence:
                        frame_num = frame_data["frame"]

                        # Convert percentage values to pixels
                        x_pct = frame_data["x"]
                        y_pct = frame_data["y"]
                        w_pct = frame_data["width"]
                        h_pct = frame_data["height"]

                        x_min = (x_pct / 100.0) * video_width
                        y_min = (y_pct / 100.0) * video_height
                        box_width = (w_pct / 100.0) * video_width
                        box_height = (h_pct / 100.0) * video_height
                        x_max = x_min + box_width
                        y_max = y_min + box_height

                        # Set the video to the correct frame (0-indexed)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                        ret, frame = cap.read()
                        if not ret:
                            print(f"Warning: Could not read frame {frame_num} from {fixed_video_filename}")
                            continue

                        # Save the extracted frame
                        image_filename = f"{fixed_video_filename.split('.')[0]}_frame_{frame_num}.jpg"
                        image_path = os.path.join(EXTRACTED_FRAMES_DIR, image_filename)
                        cv2.imwrite(image_path, frame)
                        print(f"Saved {image_path}")

                        # Prepare bounding box string (round pixel values)
                        bbox_str = f"{int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}"

                        # Write the CSV row
                        writer.writerow({
                            "annotation_id": annotation_id,
                            "video": fixed_video_filename,
                            "frame": frame_num,
                            "bounding_box": bbox_str,
                            "image_path": image_path
                        })

            cap.release()

    print(f"Frame extraction complete! CSV file written to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_frames_and_save()
