"""Muscovite ('mu'), White, Powell, Holland, Johnson & Green (2014), from
tc-mp51MnNCKFMASHTO.txt (lines 555-681). See `DioctahedralMica` for the shared
site model/mass balance this and `Margarite` both use.
"""

from __future__ import annotations

import pandas as pd

from ..dioctahedral_mica import DioctahedralMica
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 592-604.
_P_BLOCK = """
p(mu)      1 1    0  4  -1  c  -1  f  -1  n   1  y

p(cel)     2 1    1  2  -1  x  -1  y
             2    0  1  1  x    0  1  1  y

p(fcel)    2 1    0  1   1  x
             2    0  1  -1  x    0  1  1  y

p(pa)      1 1    0  1  1  n

p(mat)      1 1    0  1  1  c

p(fmu)     1 1    0  1  1  f
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 641-655.
_SF_BLOCK = """
xKA        1 1    1  2  -1  c  -1  n

xNaA       1 1    0  1  1  n

xCaA       1 1    0  1  1  c

xMgM2A     2 1    1  2  -1  x  -1  y
             2    0  1  1  x    0  1  1  y

xFeM2A     2 1    0  1   1  x
             2    0  1  -1  x    0  1  1  y

xAlM2A     1 1    0  1  1  y

xAlM2B     1 1    1  1  -1  f

xFe3M2B    1 1    0  1  1  f

xSiT1      1 1    1  2  -1/2  c  -1/2  y

xAlT1      1 1    0  2  1/2  c  1/2  y
"""


class Muscovite(DioctahedralMica):
    abbreviation = "mu"
    end_member_names = ["mu", "cel", "fcel", "pa", "mat", "fmu"]

    # -- petropandas Mineral metadata (from old TC_mu) --
    n_oxygens = 13
    ideal_cations = 5
    analytical_total_range = (96.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "A", "capacity": 1.0, "priority": ["K{+}", "Na{+}", "Ca{2+}"]},
        {"name": "M2A", "capacity": 1.0, "priority": ["Mg{2+}", "Fe{2+}", "Al{3+}"]},
        {"name": "M2B", "capacity": 1.0, "priority": ["Al{3+}", "Fe{3+}"]},
        {"name": "T1", "capacity": 2.0, "priority": ["Si{4+}", "Al{3+}"]},
    ]

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xKA...xAlT1 (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_mu = Muscovite()
