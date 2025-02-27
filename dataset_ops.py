import pandas as pd
import tensorflow as tf
import csv

def parse_row(row):
    # Read the image from the image_path.
    image_data = tf.io.read_file(row["image_path"])
    image = tf.image.decode_jpeg(image_data, channels=3)
    
    # Parse the bounding box string. It is expected to be in the format "x_min,y_min,x_max,y_max"
    bbox_str = row["bounding_box"]  
    # Split the string on commas; returns a RaggedTensor.
    bbox_split = tf.strings.split(bbox_str, ",")
    # Convert the substrings to numbers (floats).
    bbox_floats = tf.strings.to_number(bbox_split, out_type=tf.float32)
    # Stack the numbers into a tensor (ensuring its type is float32).
    bbox = tf.stack([bbox_floats[0], bbox_floats[1], bbox_floats[2], bbox_floats[3]])
    bbox = tf.cast(bbox, tf.float32)  # Ensure the type is float32
    
    # Instead of using the original label, set a dummy classification value (1)
    # so that our target becomes a dictionary with two keys.
    class_label = 1
    # class_label = tf.cast(class_label, tf.float32)  
    return image, {"bbox": bbox, "label": class_label}

def get_data_entries(csv_path="labels.csv"):
    """
    Reads the CSV file and returns a tf.data.Dataset.
    
    The CSV is assumed to have the following columns:
      annotation_id, video, frame, bounding_box, image_path
    where 'bounding_box' is a comma-separated string "x_min,y_min,x_max,y_max".
    
    This function:
      1. Reads the CSV using Pandas.
      2. Creates a tf.data.Dataset from the DataFrame.
      3. Maps each row with parse_row() to load the image and bounding box,
         and to create a target dictionary with keys "classification" and "bbox".
    
    Returns:
      A tf.data.Dataset with each element as a tuple (image, target)
      where target is a dictionary {"classification": 1, "bbox": bbox}.
    """
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Create a Dataset from the DataFrame's dictionary
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    
    # Map each row using parse_row to convert the CSV fields into image and target.
    dataset = dataset.map(lambda row: parse_row(row), num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    # For testing: load the dataset and print the shape of a few samples.
    ds = get_data_entries("labels.csv")
    for img, target in ds.take(3):
        print("Image shape:", img.shape)
        # target should be a dictionary with keys "bbox" and "label"
        print("BBox:", target["bbox"].numpy())
        print("Label:", target["label"].numpy())