"""petropandas — electron microprobe mineral analysis toolkit built on pandas."""

__version__ = "0.2.0"

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
from petropandas._plotting import ProfilePlot, ScatterPlot, TernaryPlot
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
from petropandas._series import MineralSeriesAccessor

__all__ = [
    "ALIASES",
    "Amp",
    "CationsAccessor",
    "Bt",
    "BulkAccessor",
    "Chl",
    "Cld",
    "Cpx",
    "Crd",
    "Ep",
    "Fsp",
    "Grt",
    "GrtFe3",
    "Ilm",
    "MW",
    "Mineral",
    "MineralAccessor",
    "MineralSeriesAccessor",
    "MolesAccessor",
    "Ms",
    "OxidesAccessor",
    "Opx",
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
