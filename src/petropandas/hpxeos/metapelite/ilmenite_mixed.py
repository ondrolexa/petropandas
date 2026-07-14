"""Ilmenite-hematite ('ilmm'), White, Powell, Holland, Johnson & Green (2014) + Mn
addition White, Powell & Johnson (2014), from tc-mp51MnNCKFMASHTO.txt
(lines 1751-1846). ("This model may give implausibly high Mg contents, in which case
the older, Mg,Mn-free model `Ilmenite` ('ilm') would be preferable" - axfile's own
note.)

A(Fe,Ti,Mg,Mn,Fe3+) B(Fe,Ti,Fe3+), both mult 1. Fe and Ti are shared between A and B
(not independently resolvable from bulk composition), but Mg and Mn are confined to
A alone, and - unlike other shared-cation phases in this project - Fe3+ turns out to
be tied identically between the two sites (xFe3A = xFe3B, both equal 1-i), so no
Al-style "fill one site first" convention is needed here.

i, g, m are bulk mass-balance quantities:
  g = Mg_total   (Mg confined to A, mult 1, so no division needed - g = xMgA directly)
  m = Mn_total   (Mn confined to A, mult 1, likewise)
  i = 1 - Fe3+_total / 2   (Fe3+_total = xFe3A + xFe3B = 2*(1-i) since both site
                            fractions equal 1-i; defaults to i=1 if Fe3+ not analyzed)

Q = xFeA - xFeB describes how strongly Fe/Ti order between A and B (Q=0 recovers the
disordered end-member "dilm", Q=1 the fully-ordered "oilm", matching the axfile's own
check values) - not recoverable from bulk composition, so - per the project's
convention for order-disorder phases - it is an optional caller-supplied input
defaulting to 0.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1797-1805.
_P_BLOCK = """
p(oilm)    1 1    0  1  1  Q

p(dilm)    1 1    0  4  -1  g   1  i  -1  m  -1  Q

p(dhem)    1 1    1  1  -1  i

p(geik)    1 1    0  1  1  g

p(pnt)     1 1    0  1  1  m
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1822-1836.
_SF_BLOCK = """
xFeA       1 1    0  4  -1/2  g  1/2  i  -1/2  m  1/2  Q

xTiA       1 1    0  4  -1/2  g  1/2  i  -1/2  m  -1/2  Q

xMgA       1 1    0  1  1  g

xMnA       1 1    0  1  1  m

xFe3A      1 1    1  1  -1  i

xFeB       1 1    0  4  -1/2  g  1/2  i  -1/2  m  -1/2  Q

xTiB       1 1    0  4  1/2  g  1/2  i  1/2  m  1/2  Q

xFe3B      1 1    1  1  -1  i
"""


class IlmeniteMixed(Phase):
    abbreviation = "ilmm"
    sites = {
        "A": ["Fe{2+}", "Ti{4+}", "Mg{2+}", "Mn{2+}", "Fe{3+}"],
        "B": ["Fe{2+}", "Ti{4+}", "Fe{3+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["oilm", "dilm", "dhem", "geik", "pnt"]
    order_parameter_names = ("Q",)

    # -- petropandas Mineral metadata (from old TC_ilmm) --
    n_oxygens = 3
    ideal_cations = 2
    analytical_total_range = (99.0, 101.0)
    valence_splits = []
    site_definitions = [
        {
            "name": "A",
            "capacity": 1.0,
            "priority": ["Fe{2+}", "Ti{4+}", "Mg{2+}", "Mn{2+}", "Fe{3+}"],
        },
        {"name": "B", "capacity": 1.0, "priority": ["Fe{2+}", "Ti{4+}", "Fe{3+}"]},
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        return pd.DataFrame(
            {
                "Mg": composition["Mg{2+}"],
                "Mn": composition["Mn{2+}"],
                "Fe3": fe3,
            },
            index=composition.index,
        )

    def variables(
        self,
        site_fractions: pd.DataFrame,
        order_parameters: OrderParameters | None = None,
    ) -> pd.DataFrame:
        order = resolve_order_parameters(
            order_parameters, self.order_parameter_names, site_fractions.index
        )
        return pd.DataFrame(
            {
                "i": 1.0 - site_fractions["Fe3"] / 2,
                "g": site_fractions["Mg"],
                "m": site_fractions["Mn"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xFeA...xFe3B (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_ilmm = IlmeniteMixed()
