"""Cordierite ('cd'), White, Powell, Holland, Johnson & Green (2014) + Mn addition
White, Powell & Johnson (2014), from tc-mp51MnNCKFMASHTO.txt (lines 1143-1213).

X(Fe,Mg,Mn) mult 2 is the only cation-bearing mixing site; H(H2O,vacancy) mult 1 is
the channel/cavity site that holds either a structural H2O molecule or a vacancy -
it carries no cation, so isn't part of `sites`/composition allocation.

x, m are bulk mass-balance quantities recoverable straight from composition (Fe, Mg,
Mn only ever occupy X):
  x = Fe2+_total / (Fe2+_total + Mg_total)
  m = xMnX = Mn_total / (Fe2+_total + Mg_total + Mn_total)

h (channel H2O content) is not recoverable from a cation-only composition at all (it
reflects structural water, not measured by a cation formula) - unlike Q/QAl/Q1/Q4 etc.
in other phases it isn't a true order-disorder parameter, but it's supplied through
the same optional-input mechanism (`order_parameters`), defaulting to 0 (anhydrous),
since composition alone gives no basis for a different default.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1182-1190.
_P_BLOCK = """
p(crd)      2 1    1  3  -1  h  -1  m  -1  x
              2    0  1  1  m    0  1  1  x

p(fcrd)     2 1    0  1   1  x
              2    0  1  -1  m    0  1  1  x

p(hcrd)     1 1    0  1  1  h

p(mncd)     1 1    0  1  1  m
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1203-1213.
_SF_BLOCK = """
xFeX       2 1    0  1   1  x
             2    0  1  -1  m    0  1  1  x

xMgX       2 1    1  2  -1  m  -1  x
             2    0  1  1  m    0  1  1  x

xMnX       1 1    0  1  1  m

xH2OH      1 1    0  1  1  h

xvH        1 1    1  1  -1  h
"""


class Cordierite(Phase):
    abbreviation = "cd"
    sites = {"X": ["Fe{2+}", "Mg{2+}", "Mn{2+}"]}
    end_member_names = ["crd", "fcrd", "hcrd", "mncd"]
    order_parameter_names = ("h",)

    # -- petropandas Mineral metadata (from old TC_cd) --
    n_oxygens = 18
    ideal_cations = None
    analytical_total_range = (97.0, 101.0)
    valence_splits = []
    site_definitions = [
        {"name": "X", "capacity": 2.0, "priority": ["Fe{2+}", "Mg{2+}", "Mn{2+}"]},
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        totals = self.site_totals(composition)
        return pd.DataFrame(
            {
                "xFeX": composition["Fe{2+}"] / totals["X"],
                "xMgX": composition["Mg{2+}"] / totals["X"],
                "xMnX": composition["Mn{2+}"] / totals["X"],
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
                "x": site_fractions["xFeX"]
                / (site_fractions["xFeX"] + site_fractions["xMgX"]),
                "m": site_fractions["xMnX"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xFeX, xMgX, xMnX, xH2OH, xvH (the axfile's sf block). Not used by
        `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_cd = Cordierite()
