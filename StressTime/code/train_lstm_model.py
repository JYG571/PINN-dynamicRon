# train_lstm_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dataset_utils import parse_static_rons, build_combined_dataset
from model import build_lstm_model
from physics_model import get_physics_prediction


train_file = "train.csv"
epochs = 400
batch_size = 8

print("Loading training dataset...")
df = pd.read_csv(train_file)
ron_map = parse_static_rons(df)

X_train, Y_combined_train = build_combined_dataset(df, ron_map, get_physics_prediction)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X_train)
Y_scaled = scaler_y.fit_transform(Y_combined_train)
X_scaled = np.expand_dims(X_scaled, axis=1)

model = build_lstm_model()

history = model.fit(X_scaled, Y_scaled, epochs=epochs, batch_size=batch_size, verbose=1)
