"""Biotite ('bi_G25'), Green, Holland, Powell, Weller & Riel (2025), from
tc-ig51NCKFMASHTOCr.txt. Despite the different citation, this is the Mn-free core of
`hpxeos.metapelite.Biotite` (White et al. 2014 + Mn addition) - verified
algebraically: substituting m=0 into the metapelite p-block/sf-block polynomials
reduces every term exactly to this file's (only the Al end-member's letter code
differs: `est` here vs `east` there, the axfile's own naming - kept as-is, same
"keep the axfile's own end-member code" precedent as metabasite Muscovite's `mam`).

M3(Mg,Fe,Fe3+,Ti,Al) M12(Mg,Fe) T(Si,Al), M3 mult 1, M12/T mult 2. T has a fixed
2-Si tetrahedral component outside the mixing model (total tetrahedral cations
T + fixed = 4, the same "AlIV = 4 - Si" convention as muscovite/chlorite).

x, y, f, t are bulk mass-balance quantities recoverable straight from composition:
  x = Fe2+_total / (Fe2+_total + Mg_total)   (Fe, Mg only ever occupy M3/M12)
  f = Fe3+_total                             (Fe3+ confined to M3, mult 1; defaults
                                               to 0 if not analyzed)
  t = Ti_total                               (Ti confined to M3, mult 1)
  y = Al_total - (4 - Si_total)              (Al only ever occupies M3 or T; AlIV by
                                               the standard "fill T with Si first"
                                               convention, remainder is octahedral Al
                                               = xAlM3 = y directly, since M3 mult 1)

Q describes how strongly Fe/Mg partition between M12 and M3 - not recoverable from
bulk composition, so it is an optional caller-supplied input defaulting to 0.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_P_BLOCK = """
p(phl)     4 1    1  5  -1  f  -1  t  -1  x  -1  y  -2/3  Q
             2    0  1  1  f    0  1  1  x
             2    0  1  1  t    0  1  1  x
             2    0  1  1  x    0  1  1  y

p(annm)    1 1    0  2  -1/3  Q   1  x

p(obi)     4 1    0  1   1  Q
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  t    0  1  1  x
             2    0  1  -1  x    0  1  1  y

p(est)     1 1    0  1  1  y

p(tbi)     1 1    0  1  1  t

p(fbi)     1 1    0  1  1  f
"""

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_SF_BLOCK = """
xMgM3      4 1    1  5  -1  f  -1  t  -1  x  -1  y  -2/3  Q
             2    0  1  1  f    0  1  1  x
             2    0  1  1  t    0  1  1  x
             2    0  1  1  x    0  1  1  y

xFeM3      4 1    0  2   1  x  2/3  Q
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  t    0  1  1  x
             2    0  1  -1  x    0  1  1  y

xFe3M3     1 1    0  1  1  f

xTiM3      1 1    0  1  1  t

xAlM3      1 1    0  1  1  y

xMgM12     1 1    1  2  1/3  Q  -1  x

xFeM12     1 1    0  2  -1/3  Q   1  x

xSiT       1 1    1/2  2  -1/2  f  -1/2  y

xAlT       1 1    1/2  2  1/2  f  1/2  y
"""


class Biotite(Phase):
    abbreviation = "bi_G25"
    sites = {
        "M3": ["Mg{2+}", "Fe{2+}", "Fe{3+}", "Ti{4+}", "Al{3+}"],
        "M12": ["Mg{2+}", "Fe{2+}"],
        "T": ["Si{4+}", "Al{3+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["phl", "annm", "obi", "est", "tbi", "fbi"]
    order_parameter_names = ("Q",)

    # -- petropandas Mineral metadata (from old TC_bi) --
    n_oxygens = 11
    ideal_cations = None
    analytical_total_range = (94.0, 97.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 4.0, "priority": ["Si{4+}", "Al{3+}"]},
        {"name": "I", "capacity": 1.0, "priority": ["K{+}", "Na{+}"]},
        {
            "name": "O",
            "capacity": 3.0,
            "priority": ["Mg{2+}", "Fe{2+}", "Al{3+}", "Ti{4+}", "Mn{2+}"],
        },
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        al_t = 4.0 - composition["Si{4+}"]
        return pd.DataFrame(
            {
                "Fe": composition["Fe{2+}"],
                "Mg": composition["Mg{2+}"],
                "Ti": composition["Ti{4+}"],
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
                "t": site_fractions["Ti"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM3...xAlT (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_bi_G25 = Biotite()
