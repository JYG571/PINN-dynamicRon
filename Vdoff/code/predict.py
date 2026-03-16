# predict.py

import numpy as np

from physics_model import physics_model
from config import DEFAULT_PHYSICS_PARAMS


def predict(model, scaler_x, scaler_y, Vdoff, StaticRon, Vgs=5.0, Omega=100):

    physics_pred = physics_model((Vgs, Vdoff, Omega), DEFAULT_PHYSICS_PARAMS)

    X = np.column_stack([Vdoff, np.full_like(Vdoff, StaticRon), physics_pred])
    X_norm = scaler_x.transform(X)

    residual_norm = model.predict(X_norm)
    residual = scaler_y.inverse_transform(residual_norm)

    final_pred = physics_pred + residual

    return final_pred