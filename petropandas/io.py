import numpy as np
import pandas as pd


def read_actlabs(src, **kwargs):
    """Read ActLabs Excel assay report.

    Args:
        src: Path to Excel file.

    Keyword Args:
        skiprows (int): Rows to skip before header. Default 2.

    Returns:
        tuple: (data DataFrame, units Series, limits Series, method Series)
    """
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
    """Read Bureau Veritas Excel assay report.

    Args:
        src: Path to Excel file.

    Returns:
        tuple: (data DataFrame, units Series, limits Series, method Series)
    """
    df = pd.read_excel(src, skiprows=9)
    method = df.iloc[0][2:]
    cols = df.iloc[1].copy()
    cols[:2] = ["Sample", "Type"]
    units = df.iloc[2][2:]
    limits = df.iloc[3][2:]
    selection = df.iloc[:, 1] == "Rock Pulp"

    dt = df.iloc[:, 2:][selection].copy()
    # replace detection limits
    for col in dt:
        mask = dt[col].astype(str).str.startswith("<")
        if mask.any():
            dt.loc[mask, col] = np.nan

    dt = dt.astype(float)

    res = pd.concat([df[selection].iloc[:, :2].copy(), dt], axis=1)
    res.columns = cols.str.strip()
    return res, units, limits, method
