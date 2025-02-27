import cv2

def draw_first_bbox(image_paths, bounding_boxes, classes):
    """
    Draws the bounding box for the first element of the provided arrays.

    Parameters:
      image_paths (list of str): List of image file paths.
      bounding_boxes (list of list): Each element is a list of four numbers [x_min, y_min, x_max, y_max].
      classes (list of str): List of class names corresponding to each image.
    """
    if not image_paths or not bounding_boxes or not classes:
        print("One or more input arrays are empty.")
        return

    # Get the first elements from each array.
    image_path = image_paths[0]
    bbox = bounding_boxes[0]
    cls = classes[0]

    # Read the image.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Unpack bounding box and convert to integers.
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Draw the bounding box.
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Put the class label above the bounding box.
    cv2.putText(image, cls, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the image.
    cv2.imshow("Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    # Replace with your actual arrays
    image_paths = ["dataset/Baseline_frame_1.jpg"]
    bounding_boxes = [[418, 111, 598, 411]]  # [x_min, y_min, x_max, y_max]
    classes = ["Rat"]
    
    draw_first_bbox(image_paths, bounding_boxes, classes)
