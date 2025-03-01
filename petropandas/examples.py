import importlib.resources

import pandas as pd

pkg_dir = importlib.resources.files("petropandas")

mnz = pd.read_csv(pkg_dir.joinpath("data", "mnz", "sbdata.csv")).petro.fix_columns("SB")
bulk = pd.read_csv(pkg_dir.joinpath("data", "oxides", "bulk.csv"))
