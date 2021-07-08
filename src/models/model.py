import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from utils.config import *


def get_model():
    mobile_net_model = MobileNetV2(
        include_top=False,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    model = tf.keras.models.Sequential()
    model.add(mobile_net_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(4, name='bounding_box_detector'))
    return model
