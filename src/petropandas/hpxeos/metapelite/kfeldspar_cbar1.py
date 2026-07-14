"""Ternary K-feldspar 'Cbar1 ASF' ('ksp'), Holland & Powell (2003), from
tc-mp51MnNCKFMASHTO.txt (lines 318-378). Superseded by the newer `KFeldspar`
("k4tr") model, but kept for compatibility with older THERMOCALC datasets.

Same single mixing site A(K,Na,Ca), mult 1, as `PlagioclaseCbar1` ("plc") - just
reparameterized around na = xNa, ca = xCa (K-feldspar-friendly, analogous to how
`KFeldspar`/`Plagioclase` reparameterize the same underlying "4TR" model). No Al/Si
site is tracked in this older model. No hidden order-disorder parameter.
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 347-351.
_P_BLOCK = """
p(san)     1 1    1  2  -1  ca  -1  na

p(abh)     1 1    0  1  1  na

p(anC)     1 1    0  1  1  ca
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 364-368.
_SF_BLOCK = """
xK         1 1    1  2  -1  ca  -1  na

xNa        1 1    0  1  1  na

xCa        1 1    0  1  1  ca
"""


class KFeldsparCbar1(Phase):
    abbreviation = "ksp"
    sites = {"A": ["Na{+}", "Ca{2+}", "K{+}"]}
    end_member_names = ["san", "abh", "anC"]

    # -- petropandas Mineral metadata (NEW: same formula as KFeldspar, C-1 parameterisation) --
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
                "xNa": composition["Na{+}"] / totals["A"],
                "xCa": composition["Ca{2+}"] / totals["A"],
                "xK": composition["K{+}"] / totals["A"],
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {"na": site_fractions["xNa"], "ca": site_fractions["xCa"]},
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xK, xNa, xCa (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_ksp = KFeldsparCbar1()
