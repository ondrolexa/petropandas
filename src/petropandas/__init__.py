"""petropandas — electron microprobe mineral analysis toolkit built on pandas."""

__version__ = "0.2.0"

import functools

import pandas as pd

from petropandas._accessors import (
    BulkAccessor,
    CationsAccessor,
    MineralAccessor,
    MolesAccessor,
    OxidesAccessor,
)
from petropandas._config import PPConfig, ppconfig
from petropandas._core import ALIASES, MW
from petropandas._database import PetroDB
from petropandas._minerals import (
    Amp,
    Bt,
    Chl,
    Cld,
    Cpx,
    Crd,
    Ep,
    Fsp,
    Grt,
    GrtFe3,
    Ilm,
    Mineral,
    Ms,
    Opx,
    Spl,
    St,
    Ttn,
)
from petropandas._plotting import ProfilePlot, ScatterPlot, TernaryPlot
from petropandas._series import MineralSeriesAccessor

__all__ = [
    "ALIASES",
    "MW",
    "Amp",
    "Bt",
    "BulkAccessor",
    "CationsAccessor",
    "Chl",
    "Cld",
    "Cpx",
    "Crd",
    "Ep",
    "Fsp",
    "Grt",
    "GrtFe3",
    "Ilm",
    "Mineral",
    "MineralAccessor",
    "MineralSeriesAccessor",
    "MolesAccessor",
    "Ms",
    "Opx",
    "OxidesAccessor",
    "PPConfig",
    "PetroDB",
    "ProfilePlot",
    "ScatterPlot",
    "Spl",
    "St",
    "TernaryPlot",
    "Ttn",
    "pd",
    "ppconfig",
]

# Intercept and inject engine="calamine" as the default
pd.read_excel = functools.partial(pd.read_excel, engine="calamine")
