"""Sapphirine ('sa'), Wheller & Powell (2014), from tc-mp51MnNCKFMASHTO.txt
(lines 1036-1128).

M3(Mg,Fe,Fe3+,Al) mult 1, M456(Mg,Fe) mult 3, T(Si,Al) mult 1, all per a 14-cation
(20-oxygen) formula unit that also carries a fixed, non-mixing M12-like octahedral
component (always 8 Al) and a fixed tetrahedral component (always 1 Si) - neither
varies between end-members (e.g. spr4 Mg4Al8Si2O20: M3 Mg1 + M456 Mg3 = Mg4; total Al
= 8(fixed) + 0(M3) + 0(T); total Si = 1(fixed) + 1(T)), so real compositions are
assumed normalized to this same 14-cation basis.

x, y, f are bulk mass-balance quantities recoverable straight from composition:
  x = Fe2+_total / (Fe2+_total + Mg_total)   (Fe, Mg only ever occupy M3/M456)
  f = Fe3+_total                             (Fe3+ confined to M3, mult 1; defaults
                                               to 0 if not analyzed)
  y = Al_total + Si_total - 10               (Al only ever occupies M3, T, or the
                                               fixed 8-Al component; AlT_mixing by the
                                               standard "fill T with Si first"
                                               convention over the 2 total tetrahedral
                                               cations (T mult 1 + fixed 1) gives
                                               AlT_mixing = 2 - Si_total, so
                                               y = xAlM3 = Al_total - 8 - (2-Si_total))

Q describes how strongly Fe/Mg partition between M3 and M456 - not recoverable from
bulk composition, so - per the project's convention for order-disorder phases - it is
an optional caller-supplied input defaulting to 0.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1070-1082.
_P_BLOCK = """
p(spr4)    1 1    1  4  -1/4  Q  -1  f  -1  x  -1  y

p(spr5)    1 1    0  1  1  y

p(fspm)    3 1    0  2   1  x  -3/4  Q
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  x    0  1  1  y

p(spro)    3 1    0  1   1  Q
             2    0  1  1  f    0  1  1  x
             2    0  1  1  x    0  1  1  y

p(ospr)    1 1    0  1  1  f
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1099-1117.
_SF_BLOCK = """
xMgM3      3 1    1  4  -1  f  -1  x  -1  y  3/4  Q
             2    0  1  1  f    0  1  1  x
             2    0  1  1  x    0  1  1  y

xFeM3      3 1    0  2   1  x  -3/4  Q
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  x    0  1  1  y

xFe3M3     1 1    0  1  1  f

xAlM3      1 1    0  1  1  y

xMgM456    1 1    1  2  -1/4  Q  -1  x

xFeM456    1 1    0  2  1/4  Q   1  x

xSiT       1 1    1  2  -1  f  -1  y

xAlT       1 1    0  2   1  f   1  y
"""


class Sapphirine(Phase):
    abbreviation = "sa"
    sites = {
        "M3": ["Mg{2+}", "Fe{2+}", "Fe{3+}", "Al{3+}"],
        "M456": ["Mg{2+}", "Fe{2+}"],
        "T": ["Si{4+}", "Al{3+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["spr4", "spr5", "fspm", "spro", "ospr"]
    order_parameter_names = ("Q",)

    # -- petropandas Mineral metadata (from old TC_sa) --
    n_oxygens = 20
    ideal_cations = 5
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {
            "name": "M3",
            "capacity": 1.0,
            "priority": ["Mg{2+}", "Fe{2+}", "Fe{3+}", "Al{3+}"],
        },
        {"name": "M456", "capacity": 3.0, "priority": ["Mg{2+}", "Fe{2+}"]},
        {"name": "T", "capacity": 1.0, "priority": ["Si{4+}", "Al{3+}"]},
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
                "AlOct": composition["Al{3+}"] + composition["Si{4+}"] - 10.0,
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


TC_sa = Sapphirine()
