"""Ternary plagioclase 'Ibar1 ASF' ('pli'), Holland & Powell (2003), from
tc-mb51NCKFMASHTO.txt (lines 1281-1329). "REPLACE WITH PL4TR" per the axfile's own
note - kept for compatibility with older THERMOCALC datasets, same as
`hpxeos.metapelite.plagioclase_cbar1.PlagioclaseCbar1`/`kfeldspar_cbar1.KFeldsparCbar1`
for the "Cbar1 ASF" variant.

Same single mixing site A(K,Na,Ca), mult 1, and same ca=xCa/k=xK parameterization as
`PlagioclaseCbar1`, just under different end-member names (abhI instead of abh, an
instead of anC) and a different (Ibar1, not Cbar1) asymmetric-formulation Margules
model - not needed here since only end-member proportions are computed, not
activities. No Al/Si site is tracked in this older model. No hidden order-disorder
parameter.
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 1302-1304.
_P_BLOCK = """
p(abhI)  1 1    1 2 -1 k -1 ca
p(an)    1 1    0 1  1 ca
p(san)   1 1    0 1  1 k
"""

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 1320-1322.
_SF_BLOCK = """
xK     1 1    0 1  1 k
xNa    1 1    1 2 -1 k -1 ca
xCa    1 1    0 1  1 ca
"""


class PlagioclaseIbar1(Phase):
    abbreviation = "pli"
    sites = {"A": ["Na{+}", "Ca{2+}", "K{+}"]}
    end_member_names = ["abhI", "an", "san"]

    # -- petropandas Mineral metadata (NEW: same formula as Plagioclase, I-1 parameterisation) --
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
            {"ca": site_fractions["xCa"], "k": site_fractions["xK"]},
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xK, xNa, xCa (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_pli = PlagioclaseIbar1()
