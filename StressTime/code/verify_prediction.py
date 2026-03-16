# verify_prediction.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dataset_utils import parse_static_rons
from physics_model import get_physics_prediction
from train_lstm_model import model

val_file = "verify.csv"
static_ron_list = []  # target staticRon

val_df = pd.read_csv(val_file)
val_ron_map = parse_static_rons(val_df)
val_stresst = val_df["stresst"].values.reshape(-1, 1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_val_list = []
Y_combined_list = []

for col, ron_val in val_ron_map.items():
    ron_array = np.full_like(val_stresst, ron_val)
    X_v = np.hstack([val_stresst, ron_array])
    y_true = val_df[col].values.reshape(-1,1)
    y_phys = get_physics_prediction(val_stresst.flatten(), ron_val).reshape(-1,1)
    Y_combined = np.hstack([y_true, y_phys])

    X_val_list.append(X_v)
    Y_combined_list.append(Y_combined)

X_val_all = np.vstack(X_val_list)
Y_val_combined_all = np.vstack(Y_combined_list)

pred_results = {"stresst": val_stresst.flatten()}

for target_ron in static_ron_list:
    ron_array = np.full_like(val_stresst, target_ron)
    X_val = np.hstack([val_stresst, ron_array])
    X_val_scaled = scaler_x.transform(X_val)

    Y_pred_scaled = model.predict(X_val_scaled, verbose=0)
    Y_pred_lstm = scaler_y.inverse_transform(
        np.hstack([Y_pred_scaled, np.zeros_like(Y_pred_scaled)])
    )[:,0]

    Y_pred_phys = get_physics_prediction(val_stresst.flatten(), target_ron)

    nearest_col = min(val_ron_map.keys(), key=lambda c: abs(val_ron_map[c]-target_ron))
    Y_true = val_df[nearest_col].values.flatten()

    pred_results[f"True_{nearest_col}"] = Y_true
    pred_results[f"Pred_LSTM_{target_ron:.5f}"] = Y_pred_lstm
    pred_results[f"Pred_Phys_{target_ron:.5f}"] = Y_pred_phys
