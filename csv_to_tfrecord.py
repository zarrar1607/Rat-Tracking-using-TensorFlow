import os
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util

# Adjust these paths to where your data lives
CSV_PATH = r"C:\Zarrar 2023\Programming\Rat-Tracking-using-TensorFlow\annotations.csv"
IMAGES_DIR = r"C:\Zarrar 2023\Programming\Rat-Tracking-using-TensorFlow\annotated_frames"

TRAIN_RECORD = r"C:\Zarrar 2023\Programming\Rat-Tracking-using-TensorFlow\train.record"
EVAL_RECORD = r"C:\Zarrar 2023\Programming\Rat-Tracking-using-TensorFlow\eval.record"

# Typically, you'd split your data into ~80% train, 20% eval
TRAIN_SPLIT = 0.8

def create_tf_example(row):
    # row['image_path'], row['bbox'], etc.
    image_path = row['image_path']
    full_path = os.path.join(IMAGES_DIR, image_path)

    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_image_data = fid.read()

    # If your images are .jpg, this is fine. If not, adjust accordingly.
    image_format = b'jpg'

    # Parse bounding box string
    x1, y1, x2, y2 = list(map(int, row['bbox'].split(',')))

    # You must know the width/height of your images to normalize coordinates.
    # If they are not all the same size, you'll need to read the image shape at runtime:
    # For example:
    import cv2
    img = cv2.imread(full_path)
    height, width, _ = img.shape

    # Convert to normalized [0,1]
    xmin = [x1 / width]
    xmax = [x2 / width]
    ymin = [y1 / height]
    ymax = [y2 / height]

    classes_text = [b'rat']
    classes = [1]  # Because your label map has "rat" at ID=1

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            image_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            image_path.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tfrecords():
    df = pd.read_csv(CSV_PATH)

    # Shuffle rows, then split
    df = df.sample(frac=1).reset_index(drop=True)
    train_count = int(len(df) * TRAIN_SPLIT)
    df_train = df.iloc[:train_count]
    df_eval = df.iloc[train_count:]

    # Write train.record
    with tf.io.TFRecordWriter(TRAIN_RECORD) as writer:
        for _, row in df_train.iterrows():
            tf_example = create_tf_example(row)
            writer.write(tf_example.SerializeToString())

    # Write eval.record
    with tf.io.TFRecordWriter(EVAL_RECORD) as writer:
        for _, row in df_eval.iterrows():
            tf_example = create_tf_example(row)
            writer.write(tf_example.SerializeToString())

    print(f"Created TFRecords:\n  Train: {TRAIN_RECORD}\n  Eval: {EVAL_RECORD}")

if __name__ == "__main__":
    create_tfrecords()
