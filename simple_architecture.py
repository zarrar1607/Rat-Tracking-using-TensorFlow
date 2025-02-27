import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3)):
        inputs = tf.keras.Input(shape=input_shape)

        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)

        classification_output = layers.Dense(1, activation="sigmoid", name="classification")(x)
        bbox_output = layers.Dense(4, activation="linear", name="bbox")(x)

        return models.Model(inputs=inputs, outputs=[classification_output, bbox_output])

if __name__ == "__main__":
    model = build_detection_model()
