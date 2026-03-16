# loss_function.py

import tensorflow as tf

lambda_phys = 0.05  # Changeable

def physics_informed_loss(y_true_combined, y_pred):
    y_true = y_true_combined[:, 0:1]
    y_phys = y_true_combined[:, 1:2]

    data_loss = tf.reduce_mean(tf.square(y_pred - y_true))
    phys_loss = tf.reduce_mean(tf.square(y_pred - y_phys))

    return data_loss + lambda_phys * phys_loss