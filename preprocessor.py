import tensorflow as tf

class PreProcessor:
    def __init__(self):
        pass

    def rescale(self, dataset, size=144, normalize_bbox=True):
        """
        Resizes each image in the dataset to (size, size) and scales bounding boxes accordingly.
        Optionally normalizes bounding box coordinates to [0, 1].

        Args:
          dataset: A tf.data.Dataset where each element is a tuple (image, target)
                   with target as a dictionary {"bbox": bbox, "label": label}.
          size (int): Desired size (height and width) to which each image is resized.
          normalize_bbox (bool): If True, the resulting bounding box coordinates will be normalized
                                 to the range [0, 1] relative to the image size.
        
        Returns:
          A tf.data.Dataset with resized images and updated (and optionally normalized) bounding boxes.
        """
        def resize_and_scale(image, target):
            # Directly assume target is a dictionary with keys "bbox" and "label"
            bbox = target["bbox"]
            label = target["label"]
            
            # Get original image shape (height, width)
            orig_shape = tf.cast(tf.shape(image)[:2], tf.float32)
            orig_h, orig_w = orig_shape[0], orig_shape[1]
            
            # Compute scaling factors for height and width
            scale_h = tf.cast(size, tf.float32) / orig_h
            scale_w = tf.cast(size, tf.float32) / orig_w
            
            # Resize the image to (size, size)
            image_resized = tf.image.resize(image, (size, size))
            
            # Scale bounding box coordinates to new image pixel values.
            # Original bbox is [x_min, y_min, x_max, y_max]
            x_min = bbox[0] * scale_w
            y_min = bbox[1] * scale_h
            x_max = bbox[2] * scale_w
            y_max = bbox[3] * scale_h
            bbox_scaled = tf.stack([x_min, y_min, x_max, y_max])
            
            # Optionally normalize the bounding box to [0, 1].
            if normalize_bbox:
                bbox_normalized = bbox_scaled / tf.cast(size, tf.float32)
            else:
                bbox_normalized = bbox_scaled
            
            target_updated = {"bbox": bbox_normalized, "label": label}
            return image_resized, target_updated
        
        return dataset.map(resize_and_scale, num_parallel_calls=tf.data.AUTOTUNE)

    def convert_to_grayscale(self, dataset):
        """
        Converts each image in the dataset to grayscale.
        
        Args:
          dataset: A tf.data.Dataset where each element is a tuple (image, target)
        
        Returns:
          A tf.data.Dataset where images are converted to grayscale.
        """
        def to_grayscale(image, target):
            image_gray = tf.image.rgb_to_grayscale(image)
            return image_gray, target
        
        return dataset.map(to_grayscale, num_parallel_calls=tf.data.AUTOTUNE)
    
    def split_dataset(self, dataset, train_fraction=0.8):
        """
        Splits the dataset into training and testing datasets.
        
        Args:
          dataset: A tf.data.Dataset to be split.
          train_fraction (float): Fraction of the data to use for training (default 0.8).
          
        Returns:
          A tuple (train_ds, test_ds).
        """
        total = dataset.cardinality().numpy()
        if total == tf.data.experimental.INFINITE_CARDINALITY or total == tf.data.experimental.UNKNOWN_CARDINALITY:
            raise ValueError("Dataset cardinality is unknown or infinite. Split not supported.")
        train_count = int(total * train_fraction)
        train_ds = dataset.take(train_count)
        test_ds = dataset.skip(train_count)
        return train_ds, test_ds
    
    def inspect_dataset(self, dataset, num_samples=1):
        """
        Iterates over the dataset for a few samples and prints:
          - The total number of samples (if available).
          - The image shape.
          - The target (bounding box and label) details.
        
        Args:
          dataset: A tf.data.Dataset where each element is (image, target).
          num_samples: Number of samples to inspect.
        """
        print("Inspecting dataset...")
        try:
            total = dataset.cardinality().numpy()
            print(f"Dataset cardinality: {total}")
        except Exception as e:
            print("Could not determine dataset cardinality:", e)
        
        for idx, (img, target) in enumerate(dataset.take(num_samples)):
            print(f"Sample {idx+1}:")
            print("  Image shape:", img.shape)
            bbox = target.get("bbox")
            label = target.get("label")
            print("  BBox tensor:", bbox)
            try:
                print("  BBox value:", bbox.numpy())
            except Exception as e:
                print("  BBox value: Could not convert to numpy:", e)
            print("  Label:", label)
            try:
                print("  Label value:", label.numpy())
            except Exception as e:
                print("  Label value: Could not convert to numpy:", e)
            print("-" * 40)

if __name__ == "__main__":
    # For testing: create a dummy dataset with multiple samples.
    import numpy as np
    dummy_images = []
    dummy_bboxes = []
    dummy_labels = []
    for _ in range(5):
        dummy_image = tf.convert_to_tensor(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8))
        dummy_bbox = tf.convert_to_tensor([50.0, 60.0, 250.0, 260.0])
        dummy_label = tf.constant("Rat")
        dummy_images.append(dummy_image)
        dummy_bboxes.append(dummy_bbox)
        dummy_labels.append(dummy_label)
    
    dummy_dataset = tf.data.Dataset.from_tensor_slices((
        dummy_images,
        {"bbox": dummy_bboxes, "label": dummy_labels}
    ))
    
    preproces = PreProcessor()
    # Rescale to 224x224 and normalize bounding boxes to [0, 1]
    resized_ds = preproces.rescale(dummy_dataset, size=224, normalize_bbox=True)
    # Convert to grayscale if desired (here we keep the images in RGB for detection)
    # gray_ds = preproces.convert_to_grayscale(resized_ds)
    
    train_ds, test_ds = preproces.split_dataset(resized_ds, train_fraction=0.8)
    
    print("Inspecting Training Dataset:")
    preproces.inspect_dataset(train_ds, num_samples=1)
    
    print("Inspecting Testing Dataset:")
    preproces.inspect_dataset(test_ds, num_samples=1)
