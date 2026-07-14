"""Ilmenite ('ilm'), White, Powell, Holland & Worley (2000), from
tc-mp51MnNCKFMASHTO.txt (lines 1861-1934). The older, simpler FTO-system model (no
Mg/Mn) - see `IlmeniteMixed` ("ilmm") for the newer Mg/Mn-bearing version, which the
axfile itself recommends preferring except where it gives implausibly high Mg.

A(Fe2+,Ti,Fe3+) B(Fe2+,Ti,Fe3+), both mult 1, both fully shared between sites. As in
`IlmeniteMixed`, Fe3+ is tied identically between A and B (xFe3A = xFe3B = 1-x):
  x = 1 - Fe3+_total / 2   (defaults to x=1 if Fe3+ not analyzed)

Q = xFe2A - xFe2B describes how strongly Fe2+/Ti order between A and B (Q=0 recovers
the disordered end-member "dilm", matching the axfile's own check value) - not
recoverable from bulk composition, so - per the project's convention for
order-disorder phases - it is an optional caller-supplied input defaulting to 0.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1893-1897.
_P_BLOCK = """
p(oilm)  1 1    0  1  1  Q

p(dilm)  1 1    0  2  1  x -1  Q

p(dhem)  1 1    1  1 -1  x
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1910-1920.
_SF_BLOCK = """
xFe2A  1 1    0  2  1/2  x  1/2  Q

xTiA   1 1    0  2  1/2  x -1/2  Q

xFe3A  1 1    1  1 -1  x

xFe2B  1 1    0  2  1/2  x -1/2  Q

xTiB   1 1    0  2  1/2  x  1/2  Q

xFe3B  1 1    1  1 -1  x
"""


class Ilmenite(Phase):
    abbreviation = "ilm"
    sites = {
        "A": ["Fe{2+}", "Ti{4+}", "Fe{3+}"],
        "B": ["Fe{2+}", "Ti{4+}", "Fe{3+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["oilm", "dilm", "dhem"]
    order_parameter_names = ("Q",)

    # -- petropandas Mineral metadata (from existing Ilm) --
    n_oxygens = 3
    ideal_cations = 2
    analytical_total_range = (93.0, 100.5)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {
            "name": "A",
            "capacity": 1.0,
            "priority": ["Fe{2+}", "Mg{2+}", "Mn{2+}", "Fe{3+}"],
        },
        {
            "name": "B",
            "capacity": 1.0,
            "priority": ["Ti{4+}", "Fe{3+}", "Al{3+}", "Cr{3+}"],
        },
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        return pd.DataFrame({"Fe3": fe3}, index=composition.index)

    def variables(
        self,
        site_fractions: pd.DataFrame,
        order_parameters: OrderParameters | None = None,
    ) -> pd.DataFrame:
        order = resolve_order_parameters(
            order_parameters, self.order_parameter_names, site_fractions.index
        )
        return pd.DataFrame(
            {"x": 1.0 - site_fractions["Fe3"] / 2, **order},
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xFe2A, xTiA, xFe3A, xFe2B, xTiB, xFe3B (the axfile's sf block). Not used
        by `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_ilm = Ilmenite()
