"""Ilmenite ('ilm_W24'), Weller, Holland, Soderman, Green, Powell, Beard & Riel
(2024), from tc-ig51NCKFMASHTOCr.txt. A different model shape from
`hpxeos.metabasite.IlmeniteMixed`/`hpxeos.metapelite.IlmeniteMixed` (White et al.
2014) - not a subset: `m` here is a genuine ratio over the whole A+B pool rather than
a direct single-site total, Mg mixes onto both sites (not confined to A alone), and
ordering is split into two independent parameters (`Q` for Fe, `Qt` for Ti) instead
of one.

A(Fe,Ti,Fe3+,Mg) B(Fe,Ti,Fe3+,Mg), both mult 1.

i, m are bulk mass-balance quantities recoverable straight from composition:
  i = 1 - Fe3+_total / 2   (Fe3+ is tied identically between A and B, both equal
                            1-i, so Fe3+_total = 2*(1-i); defaults to i=1 if Fe3+ not
                            analyzed)
  m = Mg_total / (Fe2+_total + Mg_total)   (Mg, Fe2+ both mix freely across A and B,
                                             but the axfile's own `m` is defined as a
                                             ratio of these same combined pools, so it
                                             reduces to the direct bulk ratio - Ti
                                             plays no part in `m` despite also mixing
                                             on both sites)

Q = xFeA - xFeB and Qt = -xTiA + xTiB describe how strongly Fe and Ti (independently)
order between A and B - neither is recoverable from bulk composition, so both are
optional caller-supplied inputs defaulting to 0 (disordered). Note the pure `oilm`
end-member (fully ordered: Fe entirely on A, Ti entirely on B) needs Q=1, Qt=1
overrides - only `dilm`/`dgk` (the disordered end-members) match the Q=Qt=0 default.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_P_BLOCK = """
p(oilm)    1 1    0  1  1  Q

p(dilm)    2 1    0  2   1  i  -1  Q
             2    0  1  -1  i    0  1  1  m

p(hem)     1 1    1  1  -1  i

p(ogk)     1 1    0  2  -1  Q   1  Qt

p(dgk)     2 1    0  2   1  Q  -1  Qt
             2    0  1  1  i    0  1  1  m
"""

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_SF_BLOCK = """
xFeA       2 1    0  2  1/2  i  1/2  Q
             2    0  1  -1/2  i    0  1  1  m

xTiA       1 1    0  2  1/2  i  -1/2  Qt

xFe3A      1 1    1  1  -1  i

xMgA       2 1    0  2  -1/2  Q  1/2  Qt
             2    0  1  1/2  i    0  1  1  m

xFeB       2 1    0  2  1/2  i  -1/2  Q
             2    0  1  -1/2  i    0  1  1  m

xTiB       1 1    0  2  1/2  i  1/2  Qt

xFe3B      1 1    1  1  -1  i

xMgB       2 1    0  2  1/2  Q  -1/2  Qt
             2    0  1  1/2  i    0  1  1  m
"""


class Ilmenite(Phase):
    abbreviation = "ilm_W24"
    sites = {
        "A": ["Fe{2+}", "Ti{4+}", "Fe{3+}", "Mg{2+}"],
        "B": ["Fe{2+}", "Ti{4+}", "Fe{3+}", "Mg{2+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["oilm", "dilm", "hem", "ogk", "dgk"]
    order_parameter_names = ("Q", "Qt")

    # -- petropandas Mineral metadata (from old TC_ilmm) --
    n_oxygens = 3
    ideal_cations = 2
    analytical_total_range = (99.0, 101.0)
    valence_splits = []
    site_definitions = [
        {
            "name": "A",
            "capacity": 1.0,
            "priority": ["Fe{2+}", "Ti{4+}", "Mg{2+}", "Mn{2+}", "Fe{3+}"],
        },
        {"name": "B", "capacity": 1.0, "priority": ["Fe{2+}", "Ti{4+}", "Fe{3+}"]},
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
                "Fe3": fe3,
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
                "i": 1.0 - site_fractions["Fe3"] / 2,
                "m": site_fractions["Mg"]
                / (site_fractions["Fe"] + site_fractions["Mg"]),
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xFeA...xMgB (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_ilm_W24 = Ilmenite()
