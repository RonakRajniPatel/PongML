from keras import layers
import keras
import tensorflow as tf


num_actions = 3
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00025, clipnorm=1.0)

# Using huber loss for stability
loss_function = tf.keras.losses.Huber()


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(350, 250, 1,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, kernel_size=5, strides=4, activation="relu")(inputs)

    layer2 = layers.Conv2D(48, kernel_size=5, strides=2, activation="relu")(layer1)

    layer3 = layers.MaxPooling2D(pool_size=(2, 2))(layer2)

    layer4 = layers.Conv2D(64, kernel_size=3, strides=2, activation="relu")(layer3)

    layer5 = layers.Conv2D(128, kernel_size=3, strides=1, activation="relu")(layer4)

    layer6 = layers.MaxPooling2D(pool_size=(2, 2))(layer5)

    layer7 = layers.Flatten()(layer6)

    layer8 = layers.Dense(512, activation="relu")(layer7)
    action = layers.Dense(num_actions, activation="linear")(layer8)

    return keras.Model(inputs=inputs, outputs=action)


def create_q_model_BACKUP():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(700, 500, 1,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)

    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)

    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)