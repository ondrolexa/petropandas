import importlib.resources
import pandas as pd

res = importlib.resources.files("petropandas").joinpath("data", "oxides")

src = res.joinpath("avgpelite.csv")
with importlib.resources.as_file(src) as f:
    avgpelite = pd.read_csv(f)

src = res.joinpath("bulk.csv")
with importlib.resources.as_file(src) as f:
    bulk = pd.read_csv(f)

src = res.joinpath("grt_profile.csv")
with importlib.resources.as_file(src) as f:
    grt_profile = pd.read_csv(f)

src = res.joinpath("minerals.csv")
with importlib.resources.as_file(src) as f:
    minerals = pd.read_csv(f)

src = res.joinpath("pyroxenes.csv")
with importlib.resources.as_file(src) as f:
    pyroxenes = pd.read_csv(f)
