import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# from simple_architecture import build_model
from mcunet import build_model  # assuming you want to use this architecture

class RatDetector:
    def __init__(self, 
                 csv_path="annotations.csv", 
                 images_dir=r"C:\Zarrar 2023\Programming\Rat-Tracking-using-TensorFlow\annotated_frames",
                 batch_size=8, 
                 orig_shape=(480,640,3), 
                 model_path="model.h5"):
        self.csv_path = csv_path
        self.images_dir = images_dir  # Directory where images are stored
        self.batch_size = batch_size
        self.orig_shape = orig_shape  # Fixed image size, e.g. (480,640,3)
        self.model = None
        self.model_path = model_path

    @staticmethod
    def normalize_bbox_np(bbox_abs, original_size):
        # original_size is (height, width, channels)
        orig_h, orig_w, _ = original_size
        norm_factors = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
        if isinstance(bbox_abs, tf.Tensor):
            bbox_abs = bbox_abs.numpy()
        return bbox_abs / norm_factors

    @staticmethod
    def denormalize_bbox_np(bbox_norm, target_size):
        # target_size is (height, width, channels)
        target_h, target_w, _ = target_size
        factors = np.array([target_w, target_h, target_w, target_h], dtype=np.float32)
        return bbox_norm * factors
    
    @staticmethod
    def rectify_path(image_path, images_dir):
        # Check if image_path already starts with "annotated_frames"
        cond = tf.strings.regex_full_match(image_path, r"^annotated_frames.*")
        # If cond is True, return image_path; else, join images_dir and image_path
        return tf.cond(cond,
                    lambda: image_path,
                    lambda: tf.strings.join([images_dir, image_path], separator="/"))


    def parse_row(self, row):
        # Get the image filename from the CSV (as a Tensor)
        image_path = row["image_path"]
        full_image_path = self.rectify_path(image_path, self.images_dir)
        
        image_data = tf.io.read_file(full_image_path)
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.image.resize(image, (self.orig_shape[0], self.orig_shape[1]))
        
        # Parse the bbox string; expected format: "x_min,y_min,x_max,y_max"
        bbox_str = row["bbox"]
        bbox_split = tf.strings.split(bbox_str, ",")
        bbox_floats = tf.strings.to_number(bbox_split, out_type=tf.float32)
        bbox_abs = tf.stack([bbox_floats[0], bbox_floats[1], bbox_floats[2], bbox_floats[3]])
        
        bbox_normalized = tf.py_function(
            func=lambda x: self.normalize_bbox_np(x, self.orig_shape),
            inp=[bbox_abs],
            Tout=tf.float32
        )
        bbox_normalized.set_shape([4])
        
        class_label = 1  # "rat"
        return image, {"classification": tf.cast(class_label, tf.float32), "bbox": bbox_normalized}




    def get_data_entries(self):
        """Loads the dataset from CSV and returns a tf.data.Dataset with images resized to orig_shape."""
        df = pd.read_csv(self.csv_path)
        dataset = tf.data.Dataset.from_tensor_slices(dict(df))
        dataset = dataset.map(lambda row: self.parse_row(row), num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def split_dataset(self, dataset, train_fraction=0.8):
        total = dataset.cardinality().numpy()
        train_count = int(total * train_fraction)
        train_ds = dataset.take(train_count).shuffle(100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = dataset.skip(train_count).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds, test_ds

    def build_model(self):
        """Builds and compiles the detection model using the fixed input shape."""
        self.model = build_model(input_shape=self.orig_shape)
        self.model.compile(
            optimizer="adam",
            loss={"classification": "binary_crossentropy", "bbox": "mse"},
            loss_weights={"classification": 1.0, "bbox": 1.0},
            metrics={"classification": "accuracy"},
        )
        self.model.summary()

    def train_model(self, epochs=5, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss={"classification": "binary_crossentropy", "bbox": "mse"},
            loss_weights={"classification": 1.0, "bbox": 1.0},
            metrics={"classification": "accuracy"},
        )
        dataset = self.get_data_entries()
        train_ds, test_ds = self.split_dataset(dataset)
        history = self.model.fit(train_ds, epochs=epochs, validation_data=test_ds)
        self.model.save(self.model_path)
        print("Model saved as " + self.model_path)
        return history

    def load_model(self, model_path=None):
        """Loads a pre-trained model."""
        if model_path:
            self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        return self.model

    def preprocess_image(self, image_path):
        # Ensure we use the correct image path
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        resized_image = cv2.resize(image, (self.orig_shape[1], self.orig_shape[0]))  # (width, height)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_norm, axis=0)
        return image, resized_image, image_batch

    def infer_image(self, image_path, ground_truth_bbox=None):
        if self.model is None:
            raise ValueError("Model is not loaded. Run `load_model()` or `train_model()` first.")
        orig_image, resized_image, image_batch = self.preprocess_image(image_path)
        predictions = self.model.predict(image_batch)
        classification_pred = predictions[0][0]   # Classification probability
        bbox_pred_norm = predictions[1][0]          # Predicted bbox (normalized)

        bbox_pred_pixels = self.denormalize_bbox_np(bbox_pred_norm, self.orig_shape)
        print(f"Classification Probability: {classification_pred}")
        print(f"Predicted BBox (normalized): {bbox_pred_norm}")
        print(f"Predicted BBox (pixels): {bbox_pred_pixels}")

        if ground_truth_bbox is not None:
            gt_norm = self.normalize_bbox_np(np.array(ground_truth_bbox), self.orig_shape)
            gt_pixels = self.denormalize_bbox_np(gt_norm, self.orig_shape)
            self.draw_bbox(resized_image, gt_pixels, "GT", (255, 0, 0))
            print(f"Ground Truth BBox (normalized): {gt_norm}")
            print(f"Ground Truth BBox (pixels): {gt_pixels}")
            print(f"Ground Truth BBox (original): {ground_truth_bbox}")

        self.draw_bbox(resized_image, bbox_pred_pixels, "Pred", (0, 255, 0))
        self.plot_results(orig_image, resized_image)

    def draw_bbox(self, image, bbox, label, color):
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def plot_results(self, orig_image, processed_image):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Prediction")
        axs[1].axis("off")
        plt.show()

    def get_entry(self, index=0):
        """
        Retrieves an image path and bounding box from the CSV.
        The bounding box is returned as absolute pixel coordinates.
        """
        df = pd.read_csv(self.csv_path)
        row = df.iloc[index]
        image_path =  row["image_path"]
        # Use column "bbox" now
        bbox_str = row["bbox"]
        bbox_vals = [float(x) for x in bbox_str.split(",")]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        print(f"Original image shape: {image.shape}")
        return image_path, bbox_vals

# Global function to plot and save training history
def plot_and_save_training_history(history, filename="training_history.png"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    if 'classification_loss' in history.history:
        plt.plot(history.history['classification_loss'], label='Train Classification Loss')
        plt.plot(history.history['val_classification_loss'], label='Val Classification Loss')
    if 'bbox_loss' in history.history:
        plt.plot(history.history['bbox_loss'], label='Train BBox Loss')
        plt.plot(history.history['val_bbox_loss'], label='Val BBox Loss')
    plt.title('Loss Breakdown')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Training history plot saved as {filename}")

# Example usage:
if __name__ == "__main__":
    detector = RatDetector(csv_path="annotations.csv", 
                             images_dir=r"C:\Zarrar 2023\Programming\Rat-Tracking-using-TensorFlow\annotated_frames",
                             batch_size=8, 
                             model_path="model.h5")
    detector.build_model()
    history = detector.train_model(epochs=20)
    plot_and_save_training_history(history, filename="./training_history.png")
    model = detector.load_model("model.h5")
    image_path, ground_truth_bbox = detector.get_entry(1)
    detector.infer_image(image_path, ground_truth_bbox)
