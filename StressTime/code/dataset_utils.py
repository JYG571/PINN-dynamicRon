# dataset_processing.py

import numpy as np

def parse_static_rons(df):
    ron_map = {}
    for col in df.columns[1:]:
        try:
            ron_val = float(col.split("_")[-1].replace(")", ""))
            ron_map[col] = ron_val
        except:
            continue
    return ron_map

def build_combined_dataset(df, ron_map, physics_model_func):
    X_list, Y_combined_list = [], []

    stresst = df["stresst"].values.reshape(-1, 1)

    for col, ron_val in ron_map.items():
        ron_array = np.full_like(stresst, ron_val)
        X_i = np.hstack([stresst, ron_array])

        y_true = df[col].values.reshape(-1, 1)
        y_phys = physics_model_func(stresst.flatten(), ron_val).reshape(-1, 1)

        Y_combined = np.hstack([y_true, y_phys])

        X_list.append(X_i)
        Y_combined_list.append(Y_combined)

    X = np.vstack(X_list)
    Y_combined = np.vstack(Y_combined_list)

    return X, Y_combined