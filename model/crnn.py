import json
import tensorflow as tf
from config.load_yaml import load_config

config = load_config()
print("config:", config)
print("config['data']['shape']: ",config['data']['shape'])


with open("dataset/char.json", 'r') as f:
    chardic = json.load(f)
with open("dataset/table.txt", 'w') as fw:
    for char in chardic:
        fw.write(char+'\n')
NUM_CLASSES = len(chardic) + 3

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',activation='relu',input_shape=config["data"]["shape"]),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3,padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(epsilon=1e-05,axis=1, momentum=0.1),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'),
        tf.keras.layers.ZeroPadding2D(padding=(0, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',activation='relu'),
        tf.keras.layers.BatchNormalization(epsilon=1e-05,axis=1, momentum=0.1),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',activation='relu'),
        tf.keras.layers.ZeroPadding2D(padding=(0, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=2, padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(epsilon=1e-05,axis=1, momentum=0.1),
        tf.keras.layers.Reshape((-1, 512)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, use_bias=True,recurrent_activation='sigmoid')),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, use_bias=True,recurrent_activation='sigmoid')),
        tf.keras.layers.Dense(units=NUM_CLASSES)
        ])