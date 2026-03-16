# model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from loss_function import physics_informed_loss


def build_lstm_model():

    model = Sequential([LSTM(64, activation='tanh', input_shape=(1, 2)),
        Dense(32, activation='relu'),
        Dense(1)])

    model.compile(optimizer="adam",loss=physics_informed_loss)

    return model