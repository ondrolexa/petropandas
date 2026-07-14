"""Augite ('aug'), augitic/calcic clinopyroxene, Green, White, Diener, Powell,
Holland & Palin (2016), from tc-mb51NCKFMASHTO.txt (lines 297-407). "WARNING:
Order-disorder on tet site only. DO NOT use for omphacitic, sodic cpx. DO NOT use for
coexisting sodic-calcic cpx" - axfile's own warning; see `Omphacite` ('dio') for that
case.

M1(Mg,Fe,Al,Fe3+) M2(Mg,Fe,Ca,Na) T1(Si,Al) T2(Si,Al), all mult 1 - unlike most other
phases in this project, there is no extra fixed/invariant tetrahedral component here
(T1+T2 alone account for all 2 tetrahedral cations in every end-member's formula, e.g.
di CaMgSi2O6: M1=Mg1, M2=Ca1, T1=Si1, T2=Si1, 4 total cations matching the formula
exactly).

x, y, f, z, j are bulk mass-balance quantities recoverable straight from composition:
  x = Fe2+_total / (Fe2+_total + Mg_total)   (Fe, Mg only ever occupy M1/M2)
  f = Fe3+_total                             (Fe3+ confined to M1, mult 1; defaults
                                               to 0 if not analyzed)
  z = Ca_total                               (Ca confined to M2, mult 1)
  j = Na_total                                (Na confined to M2, mult 1)
  y = xAlT1 + xAlT2 = 2 - Si_total            (Al only ever occupies M1, T1 or T2;
                                               T1+T2 total exactly 2 tetrahedral
                                               cations with no fixed component, so
                                               tetrahedral Al is 2 - Si_total by the
                                               standard "fill T with Si first"
                                               convention - y is *defined* as exactly
                                               this sum, so no further site-splitting
                                               is needed; Al on M1 is whatever's left
                                               of bulk Al and isn't independently
                                               needed by any variable here)

Qfm describes how strongly Fe/Mg partition between M1 and M2 (like orthopyroxene's Q);
Qal describes how strongly Al partitions between T1 and T2 specifically for the
ocats/dcats (ordered/disordered Ca-Tschermak) split - Qal=0 means xAlT1=xAlT2 (Al
evenly split, matching "dcats", the disordered end-member). Neither is recoverable
from bulk composition, so - per the project's convention for order-disorder phases -
both are optional caller-supplied inputs defaulting to 0.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 362-388.
_P_BLOCK = """
p(di)      1 1    0  2  -1  y   1  z

p(cenh)    5 1    1  4  -1/2  Qfm  -1  j  -1  x  -1  z
             2    0  1  1/2  j    0  1  1  Qfm
             2    0  1  1  j    0  1  1  x
             2    0  1  1/2  Qfm    0  1  1  z
             2    0  1  1  x    0  1  1  z

p(cfs)     5 1    0  2  -1/2  Qfm   1  x
             2    0  1  1/2  j    0  1  1  Qfm
             2    0  1  -1  j    0  1  1  x
             2    0  1  -1  x    0  1  1  y
             2    0  1  1/2  Qfm    0  1  1  z

p(jdm)     1 1    0  2  -1  f   1  j

p(acmm)    1 1    0  1  1  f

p(ocats)   1 1    0  1  1  Qal

p(dcats)   1 1    0  2  -1  Qal   1  y

p(fmc)     5 1    0  1   1  Qfm
             2    0  1  -1  j    0  1  1  Qfm
             2    0  1  1  x    0  1  1  y
             2    0  1  -1  Qfm    0  1  1  z
             2    0  1  -1  x    0  1  1  z
"""

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 432-470.
_SF_BLOCK = """
xMgM1      5 1    1  4  1/2  Qfm  -1  j  -1  x  -1  y
             2    0  1  -1/2  j    0  1  1  Qfm
             2    0  1  1  j    0  1  1  x
             2    0  1  1  x    0  1  1  y
             2    0  1  -1/2  Qfm    0  1  1  z

xFeM1      5 1    0  2  -1/2  Qfm   1  x
             2    0  1  1/2  j    0  1  1  Qfm
             2    0  1  -1  j    0  1  1  x
             2    0  1  -1  x    0  1  1  y
             2    0  1  1/2  Qfm    0  1  1  z

xAlM1      1 1    0  3  -1  f   1  j   1  y

xFe3M1     1 1    0  1  1  f

xMgM2      5 1    1  4  -1/2  Qfm  -1  j  -1  x  -1  z
             2    0  1  1/2  j    0  1  1  Qfm
             2    0  1  1  j    0  1  1  x
             2    0  1  1/2  Qfm    0  1  1  z
             2    0  1  1  x    0  1  1  z

xFeM2      5 1    0  2  1/2  Qfm   1  x
             2    0  1  -1/2  j    0  1  1  Qfm
             2    0  1  -1  j    0  1  1  x
             2    0  1  -1/2  Qfm    0  1  1  z
             2    0  1  -1  x    0  1  1  z

xCaM2      1 1    0  1  1  z

xNaM2      1 1    0  1  1  j

xSiT1      1 1    1  2  1/2  Qal  -1/2  y

xAlT1      1 1    0  2  -1/2  Qal  1/2  y

xSiT2      1 1    1  2  -1/2  Qal  -1/2  y

xAlT2      1 1    0  2  1/2  Qal  1/2  y
"""


class Augite(Phase):
    abbreviation = "aug"
    sites = {
        "M1": ["Mg{2+}", "Fe{2+}", "Al{3+}", "Fe{3+}"],
        "M2": ["Mg{2+}", "Fe{2+}", "Ca{2+}", "Na{+}"],
        "T": ["Si{4+}", "Al{3+}"],  # combined T1+T2 for mass balance / validation
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["di", "cenh", "cfs", "jdm", "acmm", "ocats", "dcats", "fmc"]
    order_parameter_names = ("Qfm", "Qal")

    # -- petropandas Mineral metadata (NEW: split from old monolithic TC_dio) --
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
                "Fe3": fe3,
                "Ca": composition["Ca{2+}"],
                "Na": composition["Na{+}"],
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
                "f": site_fractions["Fe3"],
                "z": site_fractions["Ca"],
                "j": site_fractions["Na"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM1...xAlT2 (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_aug = Augite()
