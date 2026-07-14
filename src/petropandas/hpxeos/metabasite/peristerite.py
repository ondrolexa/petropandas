"""Low-albite solid solution ('abc'), Holland, Green & Powell (2021), for modelling
the peristerite gap in metabasites, from tc-mb51NCKFMASHTO.txt (lines 998-1057).

A single pseudo-site A(NaSi, CaAl) mult 1: the charge-coupled NaSi <-> CaAl
(albite <-> anorthite) substitution is treated as one combined unit rather than
separate Na/Ca and Al/Si sites - there is no independent tetrahedral site in this
model, so Al/Si columns aren't needed at all, only Na/Ca.

ca = xCaA = Ca_total / (Na_total + Ca_total) directly (fraction of the measured A-site
cation sum; no vacancy, no other occupants, no order parameter).
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 1050-1051.
_P_BLOCK = """
p(abm)   1 1    1 1 -1 ca
p(anm)   1 1    0 1  1 ca
"""

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 1062-1063.
_SF_BLOCK = """
xNaA       1 1    1  1  -1  ca
xCaA       1 1    0  1  1  ca
"""


class Peristerite(Phase):
    abbreviation = "abc"
    sites = {"A": ["Na{+}", "Ca{2+}"]}
    end_member_names = ["abm", "anm"]

    # -- petropandas Mineral metadata (NEW: albite-oligoclase exsolution, no K) --
    n_oxygens = 8
    ideal_cations = None
    analytical_total_range = (98.5, 101.5)
    valence_splits = []
    site_definitions = [
        {"name": "A", "capacity": 1.0, "priority": ["Na{+}", "Ca{2+}"]},
        {"name": "T", "capacity": 4.0, "priority": ["Al{3+}", "Si{4+}"]},
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        totals = self.site_totals(composition)
        return pd.DataFrame(
            {
                "xNaA": composition["Na{+}"] / totals["A"],
                "xCaA": composition["Ca{2+}"] / totals["A"],
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        return pd.DataFrame({"ca": site_fractions["xCaA"]}, index=site_fractions.index)

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xNaA, xCaA (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_abc = Peristerite()
