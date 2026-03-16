# train_model.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from config import *
from dataset_processing import load_training_data
from physics_model import physics_model
from model import build_model


def train(train_file):

    df = load_training_data(train_file)

    vgs = df["Vgs_on"].values
    vds = df["Vdoff"].values
    omega = df["omega"].values

    y_data = df["Rratio"].values

    physics_pred = physics_model(
        (vgs, vds, omega),
        DEFAULT_PHYSICS_PARAMS
    )

    X = np.column_stack([
        df["Vdoff"].values,
        df["StaticRon"].values,
        physics_pred
    ])

    scaler_x = MinMaxScaler()

    X_norm = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()

    y_train = np.column_stack([
        y_data,
        physics_pred
    ])

    model = build_model()

    history = model.fit(

        X_norm,
        y_train,

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        validation_split=0.2

    )

    return model, scaler_x, scaler_y


if __name__ == "__main__":

    model, scaler_x, scaler_y = train("train.csv")