# loss_function.py

import tensorflow as tf
from config import LAMBDA_PHYS

def hybrid_loss(y_true, y_pred):

    y_data = y_true[:, 0:1]

    physics = y_true[:, 1:2]

    final_pred = physics + y_pred

    data_loss = tf.reduce_mean(
        tf.square(y_data - final_pred)
    )

    phys_reg = tf.reduce_mean(
        tf.square(y_pred)
    )

    return data_loss + LAMBDA_PHYS * phys_reg