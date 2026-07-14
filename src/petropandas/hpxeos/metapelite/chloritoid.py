"""Chloritoid ('ctd'), White, Powell, Holland, Johnson & Green (2014) + Mn addition
White, Powell & Johnson (2014), from tc-mp51MnNCKFMASHTO.txt (lines 1579-1665).

M1A(Al,Fe3+) mult 1/2 (per the axfile's own activity formulas, e.g.
"xAlM1A^(1/2)"), M1B(Fe,Mg,Mn) mult 1.

x, m are fractions of the measured M1B cation sum (M1B has no vacancy term, so this
is robust like garnet/cordierite):
  x = Fe2+_total / (Fe2+_total + Mg_total)
  m = Mn_total / (Fe2+_total + Mg_total + Mn_total)

f is different: M1A's mult is 1/2, not 1, so f = xFe3M1A is Fe3+_total / 0.5 (i.e.
2 * Fe3+_total), not the measured Al+Fe3+ sum - defaults to 0 if Fe3+ not analyzed.
Al is required as an input column (M1A's other mixing cation) for data completeness,
but - like epidote's/staurolite's Al - isn't independently needed by any formula
here.
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1619-1627.
_P_BLOCK = """
p(mctd)    2 1    1  3  -1  f  -1  m  -1  x
             2    0  1  1  m    0  1  1  x

p(fctd)    2 1    0  1   1  x
             2    0  1  -1  m    0  1  1  x

p(mnct)    1 1    0  1  1  m

p(ctdo)    1 1    0  1  1  f
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1640-1650.
_SF_BLOCK = """
xAlM1A     1 1    1  1  -1  f

xFe3M1A    1 1    0  1  1  f

xFeM1B     2 1    0  1   1  x
             2    0  1  -1  m    0  1  1  x

xMgM1B     2 1    1  2  -1  m  -1  x
             2    0  1  1  m    0  1  1  x

xMnM1B     1 1    0  1  1  m
"""


class Chloritoid(Phase):
    abbreviation = "ctd"
    sites = {
        "M1A": ["Al{3+}", "Fe{3+}"],
        "M1B": ["Fe{2+}", "Mg{2+}", "Mn{2+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["mctd", "fctd", "mnct", "ctdo"]

    # -- petropandas Mineral metadata (from old TC_ctd) --
    n_oxygens = 8
    ideal_cations = 2
    analytical_total_range = (97.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "M1A", "capacity": 1.0, "priority": ["Al{3+}", "Fe{3+}"]},
        {"name": "M1B", "capacity": 1.0, "priority": ["Fe{2+}", "Mg{2+}", "Mn{2+}"]},
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        m1b_total = (
            composition["Fe{2+}"] + composition["Mg{2+}"] + composition["Mn{2+}"]
        )
        return pd.DataFrame(
            {
                "xFeM1B": composition["Fe{2+}"] / m1b_total,
                "xMgM1B": composition["Mg{2+}"] / m1b_total,
                "xMnM1B": composition["Mn{2+}"] / m1b_total,
                "Fe3": fe3,
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": site_fractions["xFeM1B"]
                / (site_fractions["xFeM1B"] + site_fractions["xMgM1B"]),
                "m": site_fractions["xMnM1B"],
                "f": site_fractions["Fe3"] / 0.5,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xAlM1A, xFe3M1A, xFeM1B, xMgM1B, xMnM1B (the axfile's sf block). Not used
        by `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_ctd = Chloritoid()
