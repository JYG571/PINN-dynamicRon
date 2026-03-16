# dataset_processing.py

import re
import numpy as np
import pandas as pd


def extract_staticron_from_col(colname):
    m = re.search(r"#\d+[_\-]([0-9Ee+.\-]+)", colname)

    if m:
        try:
            return float(m.group(1))
        except:
            return None

    return None


def load_training_data(train_file):

    df = pd.read_csv(train_file)

    vcol = df.columns[0]

    device_cols = [c for c in df.columns if "rratio" in c.lower()]

    df_long = df.melt(
        id_vars=[vcol],
        value_vars=device_cols,
        var_name="Device",
        value_name="Rratio"
    )

    df_long = df_long.rename(columns={vcol: "Vdoff"})

    static_map = {}

    for c in device_cols:
        static_map[c] = extract_staticron_from_col(c)

    df_long["StaticRon"] = df_long["Device"].map(static_map)

    df_long = df_long.dropna().reset_index(drop=True)

    df_long["Vgs_on"] = 5.0
    df_long["omega"] = 100.0

    return df_long