"""Plagioclase ('pl4tr'), Holland, Green & Powell (2021) "4TR" ternary feldspar model
with plagioclase-friendly parameterisation, from tc-mp51MnNCKFMASHTO.txt (lines 129-186).

Same sites and end-members as `KFeldspar` (A(Na,Ca,K) mult 1, TB_4(Al,Si) mult 4) - just
reparameterized around ca = xCaA and k = xKA (rather than na, ca) since Na is the bulk
component for plagioclase-dominant compositions. No hidden order-disorder parameter:
ca, k are exactly recoverable from bulk composition (A site only ever holds Na, Ca, K).
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 159-161.
_P_BLOCK = """
p(ab)   1 1    1 2 -1 k -1 ca
p(an)   1 1    0 1  1 ca
p(san)  1 1    0 1  1 k
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 175-179.
_SF_BLOCK = """
xNaA       1 1    1  2  -1  ca  -1  k
xCaA       1 1    0  1  1  ca
xKA        1 1    0  1  1  k
xAlTB      1 1    1/4  1  1/4  ca
xSiTB      1 1    3/4  1  -1/4  ca
"""


class Plagioclase(Phase):
    abbreviation = "pl4tr"
    sites = {
        "A": ["Na{+}", "Ca{2+}", "K{+}"],
        "TB": ["Al{3+}", "Si{4+}"],
    }
    end_member_names = ["ab", "an", "san"]

    # -- petropandas Mineral metadata (from old TC_pl4tr) --
    n_oxygens = 8
    ideal_cations = None
    analytical_total_range = (98.5, 101.5)
    valence_splits = []
    site_definitions = [
        {"name": "A", "capacity": 1.0, "priority": ["Na{+}", "Ca{2+}", "K{+}"]},
        {"name": "T", "capacity": 4.0, "priority": ["Al{3+}", "Si{4+}"]},
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        totals = self.site_totals(composition)
        return pd.DataFrame(
            {
                "xNaA": composition["Na{+}"] / totals["A"],
                "xCaA": composition["Ca{2+}"] / totals["A"],
                "xKA": composition["K{+}"] / totals["A"],
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {"ca": site_fractions["xCaA"], "k": site_fractions["xKA"]},
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xNaA, xCaA, xKA, xAlTB, xSiTB (the axfile's sf block). Not used by
        `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_pl4tr = Plagioclase()
