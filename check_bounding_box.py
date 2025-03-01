import cv2
import pandas as pd
import os

CSV_PATH = 'labels.csv'

# Read the CSV file
df = pd.read_csv(CSV_PATH)

# Convert the DataFrame to a list of records (dictionaries)
annotations = df.to_dict('records')

# Start at the first image
index = 0

while True:
    # If no more annotations, break out of loop
    if len(annotations) == 0:
        print("No more images left to display.")
        break
    
    # Get current row
    row = annotations[index]
    
    # Extract image path
    image_path = row['image_path']  # adjust column name if needed
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        # Move forward if image file not found
        if index < len(annotations) - 1:
            index += 1
        else:
            break
        continue
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open image: {image_path}")
        if index < len(annotations) - 1:
            index += 1
        else:
            break
        continue

    # Parse bounding_box (assumed format: "xmin,ymin,xmax,ymax")
    bbox_str = row['bounding_box']
    x1, y1, x2, y2 = map(int, bbox_str.split(','))

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Optionally, draw text (e.g., frame number)
    frame_number = row['frame']
    cv2.putText(image, f"Frame: {frame_number}", (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the image
    cv2.imshow('Bounding Box Verification', image)
    
    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    # ESC to exit
    if key == 27:
        break
    
    # 'A' => Previous image
    elif key == ord('a'):
        if index > 0:
            index -= 1
        else:
            print("Already at the first image.")
    
    # 'D' => Next image
    elif key == ord('d'):
        if index < len(annotations) - 1:
            index += 1
        else:
            print("Already at the last image.")
    
    # 'X' => Delete the current frame/row from CSV
    elif key == ord('x'):
        # Remove the annotation in memory
        removed = annotations.pop(index)
        print(f"Deleted frame: {removed['frame']} (image: {removed['image_path']})")

        # Write the updated list back to CSV
        pd.DataFrame(annotations).to_csv(CSV_PATH, index=False)
        
        # Adjust index if needed
        if index >= len(annotations):
            index = len(annotations) - 1
        if index < 0:
            break  # no more data left
        continue  # re-display the new current index
    
    # Any other key => move to next image
    else:
        if index < len(annotations) - 1:
            index += 1
        else:
            print("Already at the last image.")

# Clean up windows
cv2.destroyAllWindows()
