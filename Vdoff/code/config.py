# config.py

import numpy as np

# training parameters
EPOCHS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# physics regularization
LAMBDA_PHYS = 0.1

# prediction targets
TARGET_STATIC_RONS = [0.17155, 0.1709]

# physics default parameters
DEFAULT_PHYSICS_PARAMS = np.array([
    0.06,0.03,10,2,0.74,4.7,
    0.025,45,20,0.023,
    0.74,350,0.94
])