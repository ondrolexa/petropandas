"""Clinopyroxene ('cpx_W24'), Weller, Holland, Soderman, Green, Powell, Beard & Riel
(2024), from tc-ig51NCKFMASHTOCr.txt. A different model generation from
`hpxeos.metabasite.Augite`/`hpxeos.metabasite.Omphacite` (Green et al. 2016): the
largest pyroxene model in the project, adding Cr3+/Ti4+/K+-bearing end-members
(`crdi`, `cbuf`, `kjd`) not present in either of those.

M1(Mg,Fe,Al,Fe3+,Cr,Ti) M2(Mg,Fe,Ca,Na,K) T(Si,Al), M1/M2 mult 1, T mult 2, no fixed
tetrahedral component (same shape as `hpxeos.igneous.Orthopyroxene`/
`hpxeos.metabasite.Augite`; confirmed against every end-member's formula, e.g. di
CaMgSi2O6: M1=Mg1, M2=Ca1, T=Si2, 4 total cations matching the formula).

x, y, o, n, f, t, cr, k are bulk mass-balance quantities recoverable straight from
composition:
  x = Fe2+_total / (Fe2+_total + Mg_total)   (Fe, Mg only ever occupy M1/M2)
  y = 2 - Si_total                           (Al only ever occupies M1 or T; T has no
                                               fixed component, so AlT cations =
                                               2 - Si_total directly, matching the
                                               axfile's own `y -> 2*xAlT`)
  o = 1 - Ca_total - Na_total - K_total      (o = xFeM2 + xMgM2 is M2's *combined*
                                               divalent occupancy; since M2 mult 1 and
                                               Ca/Na/K are all confined to M2, this is
                                               exactly the complement of the M2
                                               occupants that aren't Fe/Mg - the same
                                               "direct mult-based total" treatment as
                                               Augite's z/j, not a measured-sum ratio)
  n = Na_total                               (Na confined to M2, mult 1)
  f = Fe3+_total                             (Fe3+ confined to M1, mult 1; defaults
                                               to 0 if not analyzed)
  t = Ti_total                               (Ti confined to M1, mult 1)
  cr = Cr_total                              (Cr confined to M1, mult 1)
  k = K_total                                (K confined to M2, mult 1)

Q = -x + xFeM1/(xFeM1+xMgM1) describes how strongly Fe/Mg partition between M1 and
M2 - not recoverable from bulk composition, so it is an optional caller-supplied
input defaulting to 0.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_P_BLOCK = """
p(di)      1 1    1  4  -1  k  -1  n  -1  o  -1  y

p(cfs)     9 1    0  2   1  Q   1  x
             2    0  1  -1  k    0  1  1  Q
             2    0  1  -1  n    0  1  1  Q
             2    0  1  1  Q    0  1  1  t
             2    0  1  -1  k    0  1  1  x
             2    0  1  -1  n    0  1  1  x
             2    0  1  1  t    0  1  1  x
             2    0  1  -1  Q    0  1  1  y
             2    0  1  -1  x    0  1  1  y

p(cats)    1 1    0  4  -1  cr  -1  f   1  y  -2  t

p(crdi)    1 1    0  1  1  cr

p(cess)    1 1    0  1  1  f

p(cbuf)     1 1    0  1  2  t

p(jd)      1 1    0  1  1  n

p(cen)     6 1    0  2   1  o   1  Q
             2    0  1  -1  k    0  1  1  Q
             2    0  1  -1  n    0  1  1  Q
             2    0  1  1  Q    0  1  1  t
             2    0  1  -1  o    0  1  1  x
             2    0  1  -1  Q    0  1  1  y

p(cfm)     10 1    0  2  -1  x  -2  Q
             2    0  1  2  k    0  1  1  Q
             2    0  1  2  n    0  1  1  Q
             2    0  1  -2  Q    0  1  1  t
             2    0  1  1  k    0  1  1  x
             2    0  1  1  n    0  1  1  x
             2    0  1  1  o    0  1  1  x
             2    0  1  -1  t    0  1  1  x
             2    0  1  2  Q    0  1  1  y
             2    0  1  1  x    0  1  1  y

p(kjd)     1 1    0  1  1  k
"""

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_SF_BLOCK = """
xMgM1      9 1    1  6  -1  k  -1  n  -1  Q   1  t  -1  x  -1  y
             2    0  1  1  k    0  1  1  Q
             2    0  1  1  n    0  1  1  Q
             2    0  1  -1  Q    0  1  1  t
             2    0  1  1  k    0  1  1  x
             2    0  1  1  n    0  1  1  x
             2    0  1  -1  t    0  1  1  x
             2    0  1  1  Q    0  1  1  y
             2    0  1  1  x    0  1  1  y

xFeM1      9 1    0  2   1  Q   1  x
             2    0  1  -1  k    0  1  1  Q
             2    0  1  -1  n    0  1  1  Q
             2    0  1  1  Q    0  1  1  t
             2    0  1  -1  k    0  1  1  x
             2    0  1  -1  n    0  1  1  x
             2    0  1  1  t    0  1  1  x
             2    0  1  -1  Q    0  1  1  y
             2    0  1  -1  x    0  1  1  y

xAlM1      1 1    0  6  -1  cr  -1  f   1  k   1  n   1  y  -2  t

xFe3M1     1 1    0  1  1  f

xCrM1      1 1    0  1  1  cr

xTiM1      1 1    0  1  1  t

xMgM2      6 1    0  2   1  o   1  Q
             2    0  1  -1  k    0  1  1  Q
             2    0  1  -1  n    0  1  1  Q
             2    0  1  1  Q    0  1  1  t
             2    0  1  -1  o    0  1  1  x
             2    0  1  -1  Q    0  1  1  y

xFeM2      6 1    0  1  -1  Q
             2    0  1  1  k    0  1  1  Q
             2    0  1  1  n    0  1  1  Q
             2    0  1  -1  Q    0  1  1  t
             2    0  1  1  o    0  1  1  x
             2    0  1  1  Q    0  1  1  y

xCaM2      1 1    1  3  -1  k  -1  n  -1  o

xNaM2      1 1    0  1  1  n

xKM2       1 1    0  1  1  k

xSiT       1 1    1  1  -1/2  y

xAlT       1 1    0  1  1/2  y
"""


class Clinopyroxene(Phase):
    abbreviation = "cpx_W24"
    sites = {
        "M1": ["Mg{2+}", "Fe{2+}", "Al{3+}", "Fe{3+}", "Cr{3+}", "Ti{4+}"],
        "M2": ["Mg{2+}", "Fe{2+}", "Ca{2+}", "Na{+}", "K{+}"],
        "T": ["Si{4+}", "Al{3+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = [
        "di",
        "cfs",
        "cats",
        "crdi",
        "cess",
        "cbuf",
        "jd",
        "cen",
        "cfm",
        "kjd",
    ]
    order_parameter_names = ("Q",)

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
                "Ca": composition["Ca{2+}"],
                "Na": composition["Na{+}"],
                "K": composition["K{+}"],
                "Fe3": fe3,
                "Ti": composition["Ti{4+}"],
                "Cr": composition["Cr{3+}"],
                "AlT": 2.0 - composition["Si{4+}"],
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
                "y": site_fractions["AlT"],
                "o": 1.0
                - site_fractions["Ca"]
                - site_fractions["Na"]
                - site_fractions["K"],
                "n": site_fractions["Na"],
                "f": site_fractions["Fe3"],
                "t": site_fractions["Ti"],
                "cr": site_fractions["Cr"],
                "k": site_fractions["K"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM1...xAlT (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_cpx_W24 = Clinopyroxene()
