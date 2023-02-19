import pygame

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# retrieved from https://keras.io/examples/rl/deep_q_network_breakout/
num_actions = 3

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(500, 700, num_actions,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)