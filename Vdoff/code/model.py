# model.py

import tensorflow as tf
from config import LEARNING_RATE
from loss_function import hybrid_loss


def build_model():

    inputs = tf.keras.Input(shape=(3,))

    x = tf.keras.layers.Dense(64,activation="relu")(inputs)
    x = tf.keras.layers.Dense(16,activation="relu")(x)
    residual = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs,residual)

    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),loss=hybrid_loss)

    return model