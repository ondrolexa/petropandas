"""Orthopyroxene ('opx'), White, Powell, Holland, Johnson & Green (2014), from
tc-mb51NCKFMASHTO.txt (lines 700-780). The Mn-free core of
`hpxeos.metapelite.orthopyroxene.Orthopyroxene` (same reference, same site model minus
Mn/mnopx).

M1(Mg,Fe,Fe3+,Al) M2(Mg,Fe,Ca) T_2(Si,Al), M1/M2 mult 1, T mult 2 (the sole
tetrahedral site here, so AlT cations = 2 - Si_total directly).

x, y, f, c are bulk mass-balance quantities recoverable straight from composition:
  x = Fe2+_total / (Fe2+_total + Mg_total)   (Fe, Mg only ever occupy M1/M2)
  f = Fe3+_total                             (Fe3+ confined to M1, mult 1; defaults
                                               to 0 if not analyzed)
  c = Ca_total                               (Ca confined to M2, mult 1)
  y = Al_total - (2 - Si_total)              (Al only ever occupies M1 or T; AlIV by
                                               the standard "fill T with Si first"
                                               convention, remainder is octahedral Al
                                               = xAlM1 = y directly, since M1 mult 1)

Q describes how strongly Fe/Mg partition between M1 and M2 - not recoverable from bulk
composition, so - per the project's convention for order-disorder phases - it is an
optional caller-supplied input defaulting to 0 (no M1/M2 preference beyond bulk x).
Cr3+/Ti4+ present in real analyses have no end-member in this model and are excluded
from site allocation.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 718-737.
_P_BLOCK = """
p(en)      3 1    1  5  -1/2  Q  -1  c  -1  f  -1  x  -1  y
             2    0  1  1/2  c    0  1  1  Q
             2    0  1  1  c    0  1  1  x

p(fs)      4 1    0  2  -1/2  Q   1  x
             2    0  1  1/2  c    0  1  1  Q
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  x    0  1  1  y

p(fm)      5 1    0  1   1  Q
             2    0  1  -1  c    0  1  1  Q
             2    0  1  -1  c    0  1  1  x
             2    0  1  1  f    0  1  1  x
             2    0  1  1  x    0  1  1  y

p(mgts)    1 1    0  1  1  y

p(fopx)    1 1    0  1  1  f

p(odi)     1 1    0  1  1  c
"""

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 767-793.
_SF_BLOCK = """
xMgM1      4 1    1  4  1/2  Q  -1  f  -1  x  -1  y
             2    0  1  -1/2  c    0  1  1  Q
             2    0  1  1  f    0  1  1  x
             2    0  1  1  x    0  1  1  y

xFeM1      4 1    0  2  -1/2  Q   1  x
             2    0  1  1/2  c    0  1  1  Q
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  x    0  1  1  y

xFe3M1     1 1    0  1  1  f

xAlM1      1 1    0  1  1  y

xMgM2      3 1    1  3  -1/2  Q  -1  c  -1  x
             2    0  1  1/2  c    0  1  1  Q
             2    0  1  1  c    0  1  1  x

xFeM2      3 1    0  2  1/2  Q   1  x
             2    0  1  -1/2  c    0  1  1  Q
             2    0  1  -1  c    0  1  1  x

xCaM2      1 1    0  1  1  c

xAlT       1 1    0  2  1/2  f  1/2  y

xSiT       1 1    1  2  -1/2  f  -1/2  y
"""


class Orthopyroxene(Phase):
    abbreviation = "opx"
    sites = {
        "M1": ["Mg{2+}", "Fe{2+}", "Fe{3+}", "Al{3+}"],
        "M2": ["Mg{2+}", "Fe{2+}", "Ca{2+}"],
        "T": ["Si{4+}", "Al{3+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["en", "fs", "fm", "mgts", "fopx", "odi"]
    order_parameter_names = ("Q",)

    # -- petropandas Mineral metadata (from old TC_opx) --
    n_oxygens = 6
    ideal_cations = 4
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {
            "name": "M1",
            "capacity": 1.0,
            "priority": ["Mg{2+}", "Fe{2+}", "Mn{2+}", "Fe{3+}", "Al{3+}"],
        },
        {
            "name": "M2",
            "capacity": 1.0,
            "priority": ["Mg{2+}", "Fe{2+}", "Mn{2+}", "Ca{2+}"],
        },
        {"name": "T", "capacity": 2.0, "priority": ["Si{4+}", "Al{3+}"]},
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        al_t = 2.0 - composition["Si{4+}"]
        return pd.DataFrame(
            {
                "Fe": composition["Fe{2+}"],
                "Mg": composition["Mg{2+}"],
                "Ca": composition["Ca{2+}"],
                "Fe3": fe3,
                "AlOct": composition["Al{3+}"] - al_t,
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
                "y": site_fractions["AlOct"],
                "f": site_fractions["Fe3"],
                "c": site_fractions["Ca"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM1...xSiT (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_opx = Orthopyroxene()
