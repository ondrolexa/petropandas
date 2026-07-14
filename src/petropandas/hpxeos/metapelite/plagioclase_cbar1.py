"""Ternary plagioclase 'Cbar1 ASF' ('plc'), Holland & Powell (2003), from
tc-mp51MnNCKFMASHTO.txt (lines 256-313). Superseded by the newer `Plagioclase`
("pl4tr") model, but kept for compatibility with older THERMOCALC datasets.

A single mixing site A(K,Na,Ca), mult 1 - unlike `Plagioclase`/`KFeldspar`, this older
model does not track a tetrahedral (Al,Si) mixing site at all, so Al/Si are not part
of this model and are not required input columns.

No hidden order-disorder parameter: ca = xCa and k = xK are exactly recoverable from
bulk composition (the site only ever holds Na, Ca, K).
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 284-286.
_P_BLOCK = """
p(abh)    1 1    1 2 -1 k -1 ca
p(anC)    1 1    0 1  1 ca
p(san)    1 1    0 1  1 k
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 302-304.
_SF_BLOCK = """
x(K)     1 1    0 1  1 k
x(Na)    1 1    1 2 -1 k -1 ca
x(Ca)    1 1    0 1  1 ca
"""


class PlagioclaseCbar1(Phase):
    abbreviation = "plc"
    sites = {"A": ["Na{+}", "Ca{2+}", "K{+}"]}
    end_member_names = ["abh", "anC", "san"]

    # -- petropandas Mineral metadata (NEW: same formula as Plagioclase, C-1 parameterisation) --
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
        """x(K), x(Na), x(Ca) (the axfile's sf block). Not used by `proportions`, but
        a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_plc = PlagioclaseCbar1()
