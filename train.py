import tensorflow as tf
import os

from dataset_ops import get_data_entries
from preprocessor import PreProcessor
from simple_architecture import build_model

CSV_PATH = "labels.csv"
BATCH_SIZE = 1

dataset = get_data_entries(CSV_PATH)

preproces = PreProcessor()
dataset = preproces.rescale(dataset, size=224)

dataset = dataset.map(
    lambda image, target: (image, {"classification": tf.cast(target["label"], tf.float32), "bbox": target["bbox"]}),
    num_parallel_calls=tf.data.AUTOTUNE
)

print("Inspecting entire dataset before conversion and splitting:")
preproces.inspect_dataset(dataset, num_samples=0)

train_ds, test_ds = preproces.split_dataset(dataset, train_fraction=0.8)

train_ds = train_ds.shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Inspecting Training Dataset:")
preproces.inspect_dataset(train_ds, num_samples=1)
print("Inspecting Testing Dataset:")
preproces.inspect_dataset(test_ds, num_samples=0)

model = build_model(input_shape=(224, 224, 3))
model.compile(
    optimizer="adam",
    loss={
        'classification': 'binary_crossentropy',
        'bbox': 'mse'
    },
    loss_weights={
        'classification': 1.0,
        'bbox': 1.0  # Adjust these weights as needed
    },
    metrics={'classification': 'accuracy'}
)

# # 8. Train the model.
EPOCHS = 5
model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)

# # 9. Save the trained model.
MODEL_SAVE_PATH = "model.h5"
model.save(MODEL_SAVE_PATH)
print("Model saved to", MODEL_SAVE_PATH)
