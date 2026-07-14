"""Cordierite ('cd_G25'), White, Powell, Holland, Johnson & Green (2014), from
tc-ig51NCKFMASHTOCr.txt. The Mn-free core of `hpxeos.metapelite.Cordierite` (same
reference, same site model minus Mn/mncd - verified algebraically: substituting m=0
into the metapelite p-block/sf-block polynomials reduces every term exactly to this
file's).

X(Fe,Mg) mult 2 is the only cation-bearing mixing site; H(H2O,vacancy) mult 1 is the
channel/cavity site that holds either a structural H2O molecule or a vacancy - it
carries no cation, so isn't part of `sites`/composition allocation.

x = Fe2+_total / (Fe2+_total + Mg_total) is recoverable straight from composition
(Fe, Mg only ever occupy X).

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

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_P_BLOCK = """
p(crd)     1 1    1  2  -1  h  -1  x

p(fcrd)    1 1    0  1  1  x

p(hcrd)    1 1    0  1  1  h
"""

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_SF_BLOCK = """
xFeX       1 1    0  1  1  x

xMgX       1 1    1  1  -1  x

xH2OH      1 1    0  1  1  h

xvH        1 1    1  1  -1  h
"""


class Cordierite(Phase):
    abbreviation = "cd_G25"
    sites = {"X": ["Fe{2+}", "Mg{2+}"]}
    end_member_names = ["crd", "fcrd", "hcrd"]
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
            {"x": site_fractions["xFeX"], **order},
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xFeX...xvH (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_cd_G25 = Cordierite()
