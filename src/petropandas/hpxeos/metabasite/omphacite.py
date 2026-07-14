"""Omphacite ('dio'), sodic-calcic/omphacitic clinopyroxene, Green, White, Diener,
Powell, Holland & Palin (2016), from tc-mb51NCKFMASHTO.txt (lines 508-596). "Use this
model for coexisting sodic-calcic, omphacitic cpx! WARNING: No tet-site Al, unsuitable
for high temperatures" - axfile's own note; see `Augite` ('aug') for that case.

Each site is split into two ordering sub-positions with mult 1/2 each (M1m/M1a for M1,
M2c/M2n for M2), reflecting the P2/n omphacite ordering superstructure - but since
those sub-positions always appear as an exact m+a (or c+n) pair with matching
coefficients everywhere they're used, the *combined* M1 (mult 1: Mg,Fe,Fe3+,Al) and M2
(mult 1: Na,Ca) are all that's needed for bulk mass balance; the m/a - c/n split is
purely an ordering-state question (Q, Qaf, Qfm below). There is no tetrahedral mixing
site at all in this model (Si is always Si2, never substituted by Al), so Si/T-site
columns aren't part of `sites`.

x, f, j are bulk mass-balance quantities recoverable straight from composition:
  x = Fe2+_total / (Fe2+_total + Mg_total)     (Fe, Mg only ever occupy M1)
  f = Fe3+_total / (Al_total + Fe3+_total)     (Al, Fe3+ only ever occupy M1; f is a
                                                 ratio over just this trivalent
                                                 sub-pool of M1, per the axfile's own
                                                 legend, since M1 also hosts Mg/Fe)
  j = Na_total                                 (Na, Ca only ever occupy M2, mult 1;
                                                 j is *defined* as (xNaM2c+xNaM2n)/2,
                                                 which - since M2c/M2n each have mult
                                                 1/2 - is algebraically exactly the
                                                 bulk Na cation count, no ratio needed,
                                                 same direct-mult treatment as
                                                 Augite's z/j)

Q (Na ordering between M2c/M2n), Qaf (Fe3+/Al ordering between M1a/M1m), and Qfm
(Fe/Mg ordering specifically on M1a relative to bulk x) are not recoverable from bulk
composition, so - per the project's convention for order-disorder phases - all three
are optional caller-supplied inputs defaulting to 0.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 557-582.
_P_BLOCK = """
p(jd)      2 1    0  3   1  j  -1  Q  -1  Qaf
             2    0  1  -1  f    0  1  1  j

p(di)      5 1    1  4  -1  j  -1  Q   1  Qfm  -1  x
             2    0  1  -1  j    0  1  1  Qfm
             2    0  1  -1  Q    0  1  1  Qfm
             2    0  1  1  j    0  1  1  x
             2    0  1  -1  Q    0  1  1  x

p(hed)     5 1    0  2   1  Qfm   1  x
             2    0  1  -1  j    0  1  1  Qfm
             2    0  1  -1  Q    0  1  1  Qfm
             2    0  1  -1  j    0  1  1  x
             2    0  1  -1  Q    0  1  1  x

p(acmm)    2 1    0  1  -1  Qaf
             2    0  1  1  f    0  1  1  j

p(om)      1 1    0  1  2  Q

p(cfm)     4 1    0  1  -2  Qfm
             2    0  1  2  j    0  1  1  Qfm
             2    0  1  2  Q    0  1  1  Qfm
             2    0  1  2  Q    0  1  1  x

p(jac)     1 1    0  1  2  Qaf
"""

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 609-651.
_SF_BLOCK = """
xMgM1m     5 1    1  4  -1  j   1  Q   1  Qfm  -1  x
             2    0  1  -1  j    0  1  1  Qfm
             2    0  1  -1  Q    0  1  1  Qfm
             2    0  1  1  j    0  1  1  x
             2    0  1  -1  Q    0  1  1  x

xFeM1m     5 1    0  2  -1  Qfm   1  x
             2    0  1  1  j    0  1  1  Qfm
             2    0  1  1  Q    0  1  1  Qfm
             2    0  1  -1  j    0  1  1  x
             2    0  1  1  Q    0  1  1  x

xFe3M1m    2 1    0  1  -1  Qaf
             2    0  1  1  f    0  1  1  j

xAlM1m     2 1    0  3   1  j  -1  Q   1  Qaf
             2    0  1  -1  f    0  1  1  j

xMgM1a     5 1    1  4  -1  j  -1  Q  -1  Qfm  -1  x
             2    0  1  1  j    0  1  1  Qfm
             2    0  1  1  Q    0  1  1  Qfm
             2    0  1  1  j    0  1  1  x
             2    0  1  1  Q    0  1  1  x

xFeM1a     5 1    0  2   1  Qfm   1  x
             2    0  1  -1  j    0  1  1  Qfm
             2    0  1  -1  Q    0  1  1  Qfm
             2    0  1  -1  j    0  1  1  x
             2    0  1  -1  Q    0  1  1  x

xFe3M1a    2 1    0  1   1  Qaf
             2    0  1  1  f    0  1  1  j

xAlM1a     2 1    0  3   1  j   1  Q  -1  Qaf
             2    0  1  -1  f    0  1  1  j

xNaM2c     1 1    0  2   1  j  -1  Q

xCaM2c     1 1    1  2  -1  j   1  Q

xNaM2n     1 1    0  2   1  j   1  Q

xCaM2n     1 1    1  2  -1  j  -1  Q
"""


class Omphacite(Phase):
    abbreviation = "dio"
    sites = {
        "M1": ["Mg{2+}", "Fe{2+}", "Al{3+}", "Fe{3+}"],
        "M2": ["Na{+}", "Ca{2+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["jd", "di", "hed", "acmm", "om", "cfm", "jac"]
    order_parameter_names = ("Q", "Qaf", "Qfm")

    # -- petropandas Mineral metadata (from old TC_dio) --
    n_oxygens = 6
    ideal_cations = 4
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "T", "capacity": 2.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "M1",
            "capacity": 1.0,
            "priority": ["Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{3+}", "Mg{2+}", "Fe{2+}"],
        },
        {
            "name": "M2",
            "capacity": 1.0,
            "priority": ["Ca{2+}", "Na{+}", "Mn{2+}", "Fe{2+}", "Mg{2+}"],
        },
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        return pd.DataFrame(
            {
                "Fe": composition["Fe{2+}"],
                "Mg": composition["Mg{2+}"],
                "Al": composition["Al{3+}"],
                "Fe3": fe3,
                "Na": composition["Na{+}"],
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
                "x": site_fractions["Fe"]
                / (site_fractions["Fe"] + site_fractions["Mg"]),
                "f": site_fractions["Fe3"]
                / (site_fractions["Al"] + site_fractions["Fe3"]),
                "j": site_fractions["Na"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM1m...xCaM2n (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_dio = Omphacite()
