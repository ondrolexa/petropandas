import importlib.resources

import pandas as pd

res = importlib.resources.files("petropandas").joinpath("data")

src = res.joinpath("oxides", "avgpelite.csv")
with importlib.resources.as_file(src) as f:
    avgpelite = pd.read_csv(f)

src = res.joinpath("oxides", "bulk.csv")
with importlib.resources.as_file(src) as f:
    bulk = pd.read_csv(f)

src = res.joinpath("oxides", "grt_profile.csv")
with importlib.resources.as_file(src) as f:
    grt_profile = pd.read_csv(f)

src = res.joinpath("oxides", "minerals.csv")
with importlib.resources.as_file(src) as f:
    minerals = pd.read_csv(f)

src = res.joinpath("oxides", "pyroxenes.csv")
with importlib.resources.as_file(src) as f:
    pyroxenes = pd.read_csv(f)

src = res.joinpath("mnz", "sbdata.csv")
with importlib.resources.as_file(src) as f:
    mnz_sb = pd.read_csv(f)
