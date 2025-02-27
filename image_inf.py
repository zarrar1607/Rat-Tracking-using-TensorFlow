import cv2
import numpy as np
import tensorflow as tf

def load_detection_model(model_path):
    """Loads a saved detection model with two outputs: classification, bbox."""
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Reads and preprocesses an image for the detection model:
      - Loads image from disk (BGR).
      - Resizes to target_size.
      - Converts to RGB, normalizes, adds batch dim.
    
    Returns:
      orig_image: original BGR image (for reference or full-res drawing).
      resized_image: the 224×224 resized image (BGR) for drawing scaled boxes.
      image_batch: preprocessed image batch (1, target_size[0], target_size[1], 3).
    """
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    resized_image = cv2.resize(orig_image, target_size)
    # Convert to RGB, normalize
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_norm, axis=0)
    
    return orig_image, resized_image, image_batch

def draw_bbox_cv2(image, bbox, label="Box", color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on an image (BGR).
    
    bbox: [x_min, y_min, x_max, y_max] in pixel coords.
    label: text label to display.
    color: bounding box color.
    thickness: line thickness.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    cv2.putText(image, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)

if __name__ == "__main__":
    MODEL_PATH = "model.h5"
    IMAGE_PATH = "dataset/Baseline_frame_1.jpg"
    
    # Hard-coded ground-truth bounding box (normalized coords for 224×224).
    # Example: [0.489, 0.054, 0.8359, 0.4791]
    # If you want to retrieve from CSV, you can do so before this script.
    gt_bbox_normalized = [0.4890625, 0.05416667, 0.8359375, 0.4791667]
    
    # Load the model
    model = load_detection_model(MODEL_PATH)
    
    # Preprocess the image
    orig_image, resized_image, image_batch = preprocess_image(IMAGE_PATH, target_size=(224, 224))
    
    # Print ground-truth info
    print("Ground Truth BBox (normalized):", gt_bbox_normalized)
    gt_bbox_pixels = np.array(gt_bbox_normalized) * 224
    print("Ground Truth BBox (pixel coords):", gt_bbox_pixels)
    
    # Model inference
    predictions = model.predict(image_batch)
    if not (isinstance(predictions, list) and len(predictions) >= 2):
        raise ValueError("Model does not return two outputs (classification, bbox).")
    
    classification_pred = predictions[0][0]  # shape (1,) -> scalar
    bbox_pred = predictions[1][0]            # shape (4,) -> normalized coords
    print("Classification probability:", classification_pred)
    print("Predicted BBox (normalized):", bbox_pred)
    
    # Convert predicted bbox to pixel coords
    pred_bbox_pixels = bbox_pred * 224
    print("Predicted BBox (pixel coords):", pred_bbox_pixels)
    
    # Draw bounding boxes on the resized image
    # Ground Truth in red
    draw_bbox_cv2(resized_image, gt_bbox_pixels, label="GT", color=(0,0,255), thickness=2)
    # Prediction in green
    draw_bbox_cv2(resized_image, pred_bbox_pixels, label="Pred", color=(0,255,0), thickness=2)
    
    # Show the result
    cv2.imshow("Detection (GT=Red, Pred=Green)", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
