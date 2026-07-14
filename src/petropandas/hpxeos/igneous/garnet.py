"""Garnet ('g_W24'), Weller, Holland, Soderman, Green, Powell, Beard & Riel (2024),
from tc-ig51NCKFMASHTOCr.txt. A different model generation from
`hpxeos.metapelite.Garnet`/`hpxeos.metabasite.Garnet` (White et al. 2014): adds
Cr3+/Ti4+-bearing end-members (`knor` knorringite, `tig` Ti-garnet) that those models
explicitly exclude.

X3Y2Si3O12, M1 (Mg,Fe,Ca) mult 3, M2 (Al,Cr,Fe3+,Mg,Ti) mult 2.

x, c, f, cr are bulk mass-balance quantities recoverable straight from composition:
  c = Ca_total / 3     (Ca confined to M1, mult 3)
  f = Fe3+_total / 2    (Fe3+ confined to M2, mult 2; defaults to 0 if not analyzed)
  cr = Cr_total / 2     (Cr confined to M2, mult 2)

t = Ti_total / 2 is also a direct M2-confined ratio, but `tig` (the sole Ti end-member,
Mg3.5AlTi0.5Si3O12) couples Ti 1:1 with an *extra* Mg on M2 via charge balance
(Mg2+ + Ti4+ substituting for 2 Al3+) - so M2 carries `xMgM2 = t` too, not just Ti.
This means Mg is NOT confined to M1 the way it looks from the table at first glance:

  x = xFeM1/(xFeM1 + xMgM1)

is a ratio over M1 alone, but computing the M1-only Mg pool needs M2's Mg
contribution subtracted off first: M2 contributes `2*xMgM2 = 2*t = Ti_total` to the
bulk Mg total (M2 mult 2), so:

  x = Fe2+_total / (Fe2+_total + Mg_total - Ti_total)

(Fe2+ never occupies M2, so the M1 Fe2+ pool is the full bulk Fe2+_total unmodified.)
This is the "coupled substitution" case flagged for this phase: unlike a simple
shared-cation ratio, the M2 pool leaks into the M1-ratio calculation via `t`.

There is no order-disorder parameter in this model - x, c, f, cr, t are all exactly
recoverable from bulk composition.
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_P_BLOCK = """
p(py)      2 1    1  4  -1  c  -1  cr  -1  x  -4  t
             2    0  1  1  c    0  1  1  x

p(alm)     2 1    0  1   1  x
             2    0  1  -1  c    0  1  1  x

p(gr)      1 1    0  2   1  c  -1  f

p(andr)    1 1    0  1  1  f

p(knor)    1 1    0  1  1  cr

p(tig)     1 1    0  1  4  t
"""

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_SF_BLOCK = """
xMgM1      2 1    1  2  -1  c  -1  x
             2    0  1  1  c    0  1  1  x

xFeM1      2 1    0  1   1  x
             2    0  1  -1  c    0  1  1  x

xCaM1      1 1    0  1  1  c

xAlM2      1 1    1  3  -1  cr  -1  f  -2  t

xCrM2      1 1    0  1  1  cr

xFe3M2     1 1    0  1  1  f

xMgM2      1 1    0  1  1  t

xTiM2      1 1    0  1  1  t
"""


class Garnet(Phase):
    abbreviation = "g_W24"
    sites = {
        "M1": ["Mg{2+}", "Fe{2+}", "Ca{2+}"],
        "M2": ["Al{3+}", "Cr{3+}", "Fe{3+}", "Mg{2+}", "Ti{4+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["py", "alm", "gr", "andr", "knor", "tig"]

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
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        return pd.DataFrame(
            {
                "Fe": composition["Fe{2+}"],
                "Mg": composition["Mg{2+}"],
                "Ca": composition["Ca{2+}"],
                "Cr": composition["Cr{3+}"],
                "Fe3": fe3,
                "Ti": composition["Ti{4+}"],
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        mg_m1 = site_fractions["Mg"] - site_fractions["Ti"]
        return pd.DataFrame(
            {
                "x": site_fractions["Fe"] / (site_fractions["Fe"] + mg_m1),
                "c": site_fractions["Ca"] / 3,
                "f": site_fractions["Fe3"] / 2,
                "cr": site_fractions["Cr"] / 2,
                "t": site_fractions["Ti"] / 2,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM1...xTiM2 (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_g_W24 = Garnet()
