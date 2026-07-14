"""Garnet ('g'), White, Powell, Holland, Johnson & Green (2014) + Mn addition
White, Powell & Johnson (2014), from tc-mp51MnNCKFMASHTO.txt (lines 16-127).

X3Y2Si3O12, X = Mg,Fe2+,Mn,Ca and Y = Al,Fe3+. Cr3+/Ti4+ present in real
analyses have no end-member in this model and are excluded from site
allocation.

Unlike chlorite/muscovite, each site here has a disjoint set of cations, so
`site_fractions` is a direct per-site allocation from composition (no shared-element
mass balance needed), and there is no hidden order-disorder parameter to guess: x, z,
m, f are all exactly recoverable from bulk composition.
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 58-69.
_P_BLOCK = """
p(py)      3 1    1  4  -1  f  -1  m  -1  x  -1  z
             2    0  1  1  m    0  1  1  x
             2    0  1  1  x    0  1  1  z

p(alm)     3 1    0  1   1  x
             2    0  1  -1  m    0  1  1  x
             2    0  1  -1  x    0  1  1  z

p(spss)    1 1    0  1  1  m

p(gr)      1 1    0  1  1  z

p(kho)     1 1    0  1  1  f
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 92-106.
_SF_BLOCK = """
xMgX       3 1    1  3  -1  m  -1  x  -1  z
             2    0  1  1  m    0  1  1  x
             2    0  1  1  x    0  1  1  z

xFeX       3 1    0  1   1  x
             2    0  1  -1  m    0  1  1  x
             2    0  1  -1  x    0  1  1  z

xMnX       1 1    0  1  1  m

xCaX       1 1    0  1  1  z

xAlY       1 1    1  1  -1  f

xFe3Y      1 1    0  1  1  f
"""


class Garnet(Phase):
    abbreviation = "g"
    sites = {
        "X": ["Mg{2+}", "Fe{2+}", "Mn{2+}", "Ca{2+}"],
        "Y": ["Al{3+}", "Fe{3+}"],
    }
    end_member_names = ["py", "alm", "spss", "gr", "kho"]

    # -- petropandas Mineral metadata (from old TC_g) --
    n_oxygens = 12
    ideal_cations = 8
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "Z", "capacity": 3.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "Y",
            "capacity": 2.0,
            "priority": ["Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{3+}"],
        },
        {
            "name": "X",
            "capacity": 3.0,
            "priority": ["Fe{2+}", "Mg{2+}", "Ca{2+}", "Mn{2+}"],
        },
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        totals = self.site_totals(composition)
        return pd.DataFrame(
            {
                "xMgX": composition["Mg{2+}"] / totals["X"],
                "xFeX": composition["Fe{2+}"] / totals["X"],
                "xMnX": composition["Mn{2+}"] / totals["X"],
                "xCaX": composition["Ca{2+}"] / totals["X"],
                "xAlY": composition["Al{3+}"] / totals["Y"],
                "xFe3Y": composition["Fe{3+}"] / totals["Y"],
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        # x -> xFeX/(xFeX + xMgX), z -> xCaX, m -> xMnX, f -> xFe3Y
        return pd.DataFrame(
            {
                "x": site_fractions["xFeX"]
                / (site_fractions["xFeX"] + site_fractions["xMgX"]),
                "z": site_fractions["xCaX"],
                "m": site_fractions["xMnX"],
                "f": site_fractions["xFe3Y"],
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgX, xFeX, xMnX, xCaX, xAlY, xFe3Y (the axfile's sf block). Not used by
        `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_g = Garnet()
