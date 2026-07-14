"""Staurolite ('st'), White, Powell, Holland, Johnson & Green (2014), from
tc-mp51MnNCKFMASHTO.txt (lines 1233-1330).

X(Mg,Fe,Mn) mult 4, Y(Al,Fe3+,Ti,v) mult 2 - Y carries a genuine structural vacancy
(v), a well-known feature of staurolite's unusual site topology, not merely an
unmodeled trace element.

x, m are computed like garnet/cordierite (fractions of the measured X-site cation
sum, since X has no vacancy term and is always fully occupied):
  x = Fe2+_total / (Fe2+_total + Mg_total)
  m = Mn_total / (Fe2+_total + Mg_total + Mn_total)

f, t are different: because Y genuinely has vacancies, dividing by the *measured*
Al+Fe3++Ti sum (which may fall short of 2 for a real, vacancy-bearing analysis) would
incorrectly erase that vacancy. So f, t are fractions of Y's fixed multiplicity (2)
instead:
  f = xFe3Y = Fe3+_total / 2   (defaults to 0 if Fe3+ not analyzed)
  t = xTiY  = Ti_total / 2
Al is required as an input column (Y's third mixing cation, alongside Fe3+/Ti) for
data completeness, but - like epidote's Al - isn't independently needed by any
formula here, since xAlY is simply whatever isn't Fe3+, Ti, or vacant.
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1269-1279.
_P_BLOCK = """
p(mstm)    2 1    1  4  -1  f  -1  m  -1  x  -4/3  t
             2    0  1  1  m    0  1  1  x

p(fst)     2 1    0  1   1  x
             2    0  1  -1  m    0  1  1  x

p(mnstm)   1 1    0  1  1  m

p(msto)    1 1    0  1  1  f

p(mstt)    1 1    0  1  4/3  t
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1296-1310.
_SF_BLOCK = """
xMgX       2 1    1  2  -1  m  -1  x
             2    0  1  1  m    0  1  1  x

xFeX       2 1    0  1   1  x
             2    0  1  -1  m    0  1  1  x

xMnX       1 1    0  1  1  m

xAlY       1 1    1  2  -1  f  -4/3  t

xFe3Y      1 1    0  1  1  f

xTiY       1 1    0  1  1  t

xvY        1 1    0  1  1/3  t
"""


class Staurolite(Phase):
    abbreviation = "st"
    sites = {
        "X": ["Mg{2+}", "Fe{2+}", "Mn{2+}"],
        "Y": ["Al{3+}", "Fe{3+}", "Ti{4+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["mstm", "fst", "mnstm", "msto", "mstt"]

    # -- petropandas Mineral metadata (from old TC_st) --
    n_oxygens = 48
    ideal_cations = 6
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "X", "capacity": 4.0, "priority": ["Mg{2+}", "Fe{2+}", "Mn{2+}"]},
        {"name": "Y", "capacity": 2.0, "priority": ["Al{3+}", "Fe{3+}", "Ti{4+}"]},
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        # Not using self.site_totals(): it would sum Y's cations too, including the
        # optional Fe{3+}, which may not be present.
        x_total = composition["Mg{2+}"] + composition["Fe{2+}"] + composition["Mn{2+}"]
        return pd.DataFrame(
            {
                "xFeX": composition["Fe{2+}"] / x_total,
                "xMgX": composition["Mg{2+}"] / x_total,
                "xMnX": composition["Mn{2+}"] / x_total,
                "Fe3": fe3,
                "Ti": composition["Ti{4+}"],
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": site_fractions["xFeX"]
                / (site_fractions["xFeX"] + site_fractions["xMgX"]),
                "m": site_fractions["xMnX"],
                "f": site_fractions["Fe3"] / 2,
                "t": site_fractions["Ti"] / 2,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgX, xFeX, xMnX, xAlY, xFe3Y, xTiY, xvY (the axfile's sf block). Not used
        by `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_st = Staurolite()
