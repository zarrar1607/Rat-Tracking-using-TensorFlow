import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3)):
    inputs = keras.Input(shape=input_shape)
    # ---------------------------
    # Feature Extraction (backbone)
    # ---------------------------
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)
    x = keras.layers.Conv2D(16, (3,3), padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(8, (1,1), padding='valid')(x)
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3), padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    y = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(80, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
    # add first skip connection
    y = keras.layers.Add()([x, y])
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    x = keras.layers.DepthwiseConv2D((5,5), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
    # add second skip connection
    y = keras.layers.Add()([x, y])
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
    # add third skip connection
    x = keras.layers.Add()([x, y])
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(80, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
    x = keras.layers.DepthwiseConv2D((7,7), padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    y = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    x = keras.layers.DepthwiseConv2D((5,5), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
    # add fourth skip connection
    y = keras.layers.Add()([x, y])
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = keras.layers.DepthwiseConv2D((3,3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
    # add fifth skip connection
    y = keras.layers.Add()([x, y])
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = keras.layers.DepthwiseConv2D((7,7), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
    # add sixth skip connection
    x = keras.layers.Add()([x, y])
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.DepthwiseConv2D((3,3), padding='valid', strides=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    y = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(240, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
    x = keras.layers.DepthwiseConv2D((7,7), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
    # add seventh skip connection
    y = keras.layers.Add()([x, y])
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(160, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    x = keras.layers.DepthwiseConv2D((5,5), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
    # add eighth skip connection
    y = keras.layers.Add()([x, y])
    
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
    x = keras.layers.Conv2D(200, (1,1), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = keras.layers.DepthwiseConv2D((3,3), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(max_value=6.0)(x)
    x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    x = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
    # add ninth skip connection
    x = keras.layers.Add()([x, y])
    
    # x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    # x = keras.layers.Conv2D(200, (1,1), padding='valid')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.ReLU(max_value=6.0)(x)
    # x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
    # x = keras.layers.DepthwiseConv2D((5,5), padding='valid')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.ReLU(max_value=6.0)(x)
    # x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
    # y = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
    # # add tenth skip connection
    # x = keras.layers.Add()([x, y])
    
    # ---------------------------
    # Two-headed output branches:
    # ---------------------------
    # Global average pooling from the last feature map:
    features = keras.layers.GlobalAveragePooling2D()(x)
    
    # Classification branch:
    cls = keras.layers.Dense(64, activation="relu")(features)
    classification_output = keras.layers.Dense(1, activation="sigmoid", name="classification")(cls)
    
    # Regression branch for bounding boxes:
    reg = keras.layers.Dense(64, activation="relu")(features)
    reg = keras.layers.Dense(32, activation="relu")(reg)
    bbox_output = keras.layers.Dense(4, activation="linear", name="bbox")(reg)
    
    model = keras.models.Model(inputs=inputs, outputs=[classification_output, bbox_output])
    return model

if __name__ == "__main__":
    model = build_model(input_shape=(224,224,3))
