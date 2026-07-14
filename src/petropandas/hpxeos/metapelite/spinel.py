"""Spinel ('sp'), White, Powell & Clarke (2002), from tc-mp51MnNCKFMASHTO.txt
(lines 1671-1745). The axfile itself notes M1(Mg,Fe), M2(Al,Fe3+,Ti) are "mixing
sites (not true sites)" - conceptual cation groupings, not literal crystallographic
sites with a multiplicity to track.

x, y, z are bulk mass-balance quantities recoverable straight from composition. x is
a simple ratio (Fe2+, Mg only ever occupy "M1"):
  x = Fe2+_total / (Fe2+_total + Mg_total)
y, z are self-normalizing ratios over the (Al, Fe3+, Ti) pool, by the axfile's own
definition - so no assumption about "M2"'s absolute multiplicity is needed at all:
  y = Al_total / (Al_total + Fe3+_total + 2*Ti_total)
  z = 2*Ti_total / (Al_total + Fe3+_total + 2*Ti_total)
Fe3+ defaults to 0 if not analyzed.
"""

from __future__ import annotations

import pandas as pd

from ..base import Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1702-1709.
_P_BLOCK = """
p(herc)  2 1    0  1   1  y
             2   -1  1   1  x    1  1   1  z

p(sp)    1 2    1  1  -1  x    1  1   1  z

p(mt)    1 1    1  2  -1  y -1  z

p(usp)   1 1    0  1   1  z
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1726-1734.
_SF_BLOCK = """
x(Al)       1 1    0  1  1  y

x(Fe3)      1 1    1  2 -1  y -1  z

x(Ti)       1 1    0  1  1  z

x(Mg)       1 1    1  1 -1  x

x(Fe2)      1 1    0  1  1  x
"""


class Spinel(Phase):
    abbreviation = "sp"
    sites = {
        "M1": ["Mg{2+}", "Fe{2+}"],
        "M2": ["Al{3+}", "Fe{3+}", "Ti{4+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["herc", "sp", "mt", "usp"]

    # -- petropandas Mineral metadata (from old TC_sp / TC_Spinel) --
    n_oxygens = 4
    ideal_cations = 3
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "M1", "capacity": 1.0, "priority": ["Mg{2+}", "Fe{2+}"]},
        {"name": "M2", "capacity": 2.0, "priority": ["Al{3+}", "Fe{3+}", "Ti{4+}"]},
    ]

    def _preprocess_oxides(self, oxide_df):
        """Merge Fe2O3 into FeO before standard APFU + Droop."""
        from petropandas import _calc

        return _calc.fe2o3_to_feo(oxide_df)

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
                "Ti": composition["Ti{4+}"],
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        pool2 = site_fractions["Al"] + site_fractions["Fe3"] + 2 * site_fractions["Ti"]
        return pd.DataFrame(
            {
                "x": site_fractions["Fe"]
                / (site_fractions["Fe"] + site_fractions["Mg"]),
                "y": site_fractions["Al"] / pool2,
                "z": 2 * site_fractions["Ti"] / pool2,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """x(Al), x(Fe3), x(Ti), x(Mg), x(Fe2) (the axfile's sf block). Not used by
        `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_sp = Spinel()
