import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from simple_architecture import build_model


class RatDetector:
    def __init__(self, csv_path="labels.csv", batch_size=8, img_size=224, model_path = "model.h5"):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.model = None
        self.model_path = model_path

    def parse_row(self, row):
        """Parses a row from the CSV to extract image, bounding box, and classification."""
        image_data = tf.io.read_file(row["image_path"])
        image = tf.image.decode_jpeg(image_data, channels=3)

        # Parse bounding box (string "x_min,y_min,x_max,y_max")
        bbox_str = row["bounding_box"]
        bbox_split = tf.strings.split(bbox_str, ",")
        bbox_floats = tf.strings.to_number(bbox_split, out_type=tf.float32)
        bbox = tf.stack([bbox_floats[0], bbox_floats[1], bbox_floats[2], bbox_floats[3]])

        # Classification label (always 1 for presence of a rat)
        class_label = 1

        return image, {"classification": tf.cast(class_label, tf.float32), "bbox": bbox}

    def get_data_entries(self):
        """Loads the dataset from CSV and returns a tf.data.Dataset."""
        df = pd.read_csv(self.csv_path)
        dataset = tf.data.Dataset.from_tensor_slices(dict(df))
        dataset = dataset.map(lambda row: self.parse_row(row), num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def preprocess_dataset(self, dataset, normalize_bbox=True):
        """Rescales images and normalizes bounding boxes if needed."""
        def resize_and_scale(image, target):
            bbox = target["bbox"]
            label = target["classification"]

            # Resize image
            image_resized = tf.image.resize(image, (self.img_size, self.img_size))

            # Normalize bounding boxes
            if normalize_bbox:
                bbox = bbox / tf.cast(self.img_size, tf.float32)

            return image_resized, {"classification": label, "bbox": bbox}

        return dataset.map(resize_and_scale, num_parallel_calls=tf.data.AUTOTUNE)

    def split_dataset(self, dataset, train_fraction=0.8):
        """Splits dataset into training and testing sets."""
        total = dataset.cardinality().numpy()
        train_count = int(total * train_fraction)
        train_ds = dataset.take(train_count).shuffle(100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = dataset.skip(train_count).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds, test_ds

    def build_model(self):
        """Builds a two-headed CNN model for classification and bounding box regression."""
        self.model = build_model(input_shape=(self.img_size, self.img_size, 3))
        self.model.compile(
            optimizer="adam",
            loss={"classification": "binary_crossentropy", "bbox": "mse"},
            loss_weights={"classification": 1.0, "bbox": 1.0},
            metrics={"classification": "accuracy"},
        )
        self.model.summary()

    def train_model(self, epochs=5):
        """Trains the model on the dataset."""
        dataset = self.get_data_entries()
        dataset = self.preprocess_dataset(dataset)
        train_ds, test_ds = self.split_dataset(dataset)

        self.model.fit(train_ds, epochs=epochs, validation_data=test_ds)

        self.model.save(self.model_path)
        print("Model saved as " + self.model_path)

    def load_model(self, model_path="model.h5"):
        """Loads a pre-trained model."""
        if(model_path):
            self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def preprocess_image(self, image_path):
        """Preprocesses an image for inference."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")

        resized_image = cv2.resize(image, (self.img_size, self.img_size))
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_norm, axis=0)

        return image, resized_image, image_batch

    def infer_image(self, image_path, ground_truth_bbox=None):
        """Runs inference on a single image and plots ground truth vs. prediction."""
        if self.model is None:
            raise ValueError("Model is not loaded. Run `load_model()` or `train_model()` first.")

        orig_image, resized_image, image_batch = self.preprocess_image(image_path)
        predictions = self.model.predict(image_batch)

        classification_pred = predictions[0][0]  # Probability of presence
        bbox_pred = predictions[1][0] * self.img_size  # Convert to pixel coordinates


        print(f"Classification Probability: {classification_pred}")
        print(f"Predicted BBox (pixels): {bbox_pred}")

        # Draw ground truth (if provided)
        if ground_truth_bbox is not None:
            gt_bbox_pixels = np.array(ground_truth_bbox) * self.img_size
            self.draw_bbox(resized_image, gt_bbox_pixels, "GT", (255, 0, 0))
            print(f"Ground Truth BBox (pixels): {ground_truth_bbox}")

        # Draw prediction
        self.draw_bbox(resized_image, bbox_pred, "Pred", (0, 255, 0))

        # Show images
        self.plot_results(orig_image, resized_image)

    def draw_bbox(self, image, bbox, label, color):
        """Draws bounding box on image."""
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def plot_results(self, orig_image, processed_image):
        """Displays the original and processed image side by side."""
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Prediction")
        axs[1].axis("off")

        plt.show()

    def get_entry(self, index = 0):
        df = pd.read_csv(self.csv_path)
        row = df.iloc[index]

        image_path = row["image_path"]
        bbox_str = row["bounding_box"]
        bbox_vals = [float(x) for x in bbox_str.split(",")]

        # Read the image to get original size
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")

        orig_h, orig_w, _ = image.shape  # Get height and width
        print(image.shape)

        # Normalize bounding box coordinates
        bbox_normalized = np.array([
            bbox_vals[0] / orig_w,  # x_min
            bbox_vals[1] / orig_h,  # y_min
            bbox_vals[2] / orig_w,  # x_max
            bbox_vals[3] / orig_h   # y_max
        ])

        return image_path, bbox_normalized
    


# Example usage:
if __name__ == "__main__":
    detector = RatDetector(img_size = 144, model_path='model.h5')
    # detector.build_model()
    # detector.train_model(epochs=5)
    detector.load_model()
    image_path, ground_truth_bbox = detector.get_entry(0)
    detector.infer_image(image_path, ground_truth_bbox)
