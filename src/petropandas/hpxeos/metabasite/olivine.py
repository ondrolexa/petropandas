"""Olivine ('ol'), Holland & Powell (2011), from tc-mb51NCKFMASHTO.txt
(lines 924-950).

M(Mg,Fe) mult 2 - the only site, single-owner cations, no order parameter.

x = xFeM = Fe_total / (Fe_total + Mg_total) directly (fraction of the measured M-site
cation sum; M has no other occupants and no vacancy term).
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 934-936.
_P_BLOCK = """
p(fo)      1  1    1  1 -1  x

p(fa)      1  1    0  1  1  x
"""

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 944-946.
_SF_BLOCK = """
xMgM    1 1      1  1 -1  x

xFeM    1 1      0  1  1  x
"""


class Olivine(Phase):
    abbreviation = "ol"
    sites = {"M": ["Mg{2+}", "Fe{2+}"]}
    end_member_names = ["fo", "fa"]

    # -- petropandas Mineral metadata (NEW: standard (Mg,Fe)2SiO4 olivine formula) --
    n_oxygens = 4
    ideal_cations = 3
    analytical_total_range = (98.0, 101.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 1.0, "priority": ["Si{4+}"]},
        {
            "name": "M",
            "capacity": 2.0,
            "priority": ["Mg{2+}", "Fe{2+}", "Mn{2+}", "Ca{2+}"],
        },
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        totals = self.site_totals(composition)
        return pd.DataFrame(
            {
                "xMgM": composition["Mg{2+}"] / totals["M"],
                "xFeM": composition["Fe{2+}"] / totals["M"],
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        return pd.DataFrame({"x": site_fractions["xFeM"]}, index=site_fractions.index)

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM, xFeM (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_ol = Olivine()
