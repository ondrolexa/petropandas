import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_actlabs(src, **kwargs):
    if "skiprows" not in kwargs:
        kwargs["skiprows"] = 2
    df = pd.read_excel(src, **kwargs).rename(columns={"Fe2O3(T)": "Fe2O3"})
    units = df.iloc[0]
    limits = df.iloc[1]
    method = df.iloc[2]
    df = df.rename(columns={"Analyte Symbol": "Sample"})[3:].set_index("Sample")
    # replace detection limits
    for col in df:
        ix = df[col].astype(str).str.startswith("< ")
        if any(ix):
            df.loc[ix, col] = np.nan

    df = df.astype(float)
    return df, units, limits, method


def read_bureau_veritas(src=""):
    df = pd.read_excel(src, skiprows=9)
    method = df.iloc[0][2:]
    cols = df.iloc[1]
    cols[:2] = ["Sample", "Type"]
    units = df.iloc[2][2:]
    limits = df.iloc[3][2:]
    selection = df.iloc[:, 1] == "Rock Pulp"

    dt = df.iloc[:, 2:][selection]
    # replace detection limits
    for col in dt:
        dt[col][dt[col].astype(str).str.startswith("<") is True] = 0

    dt = dt.astype(float).copy()

    res = pd.concat([df[selection].iloc[:, :2], dt], axis=1)
    res.columns = cols.str.strip()
    return res, units, limits, method
