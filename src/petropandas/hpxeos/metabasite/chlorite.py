"""Chlorite ('chl'), White, Powell, Holland, Johnson & Green (2014), from
tc-mb51NCKFMASHTO.txt (lines 1911-2081, the last phase block in this file). The
Mn-free core of `hpxeos.metapelite.chlorite.Chlorite` (same reference, same site model
minus Mn/mmchl).

M1(Mg,Fe,Al) M23_4(Mg,Fe) M4(Mg,Fe,Fe3+,Al) T2_2(Si,Al), plus a fixed T1 site holding
2 Si always (outside the mixing model, so never referenced below).

x, y, f are bulk mass-balance quantities recoverable straight from composition
(Fe/Mg never leave the octahedral sites; Al only ever occupies M1, M4 or T2 in this
model, and AlT2 follows the standard "AlIV = 4 - Si" tetrahedral convention since
T1+T2 always total 4 tetrahedral cations). QAl, Q1, Q4 describe how Al/Fe/Mg partition
between M1/M23/M4 specifically - that ordering state is not recoverable from bulk
composition, so it is an optional caller-supplied input (`order_parameters`,
defaulting to 0 = fully disordered; see `resolve_order_parameters`).
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 1944-1993.
_P_BLOCK = """
p(clin)    9 1    0  3  -1/4  Q1  2  QAl  -5/4  Q4
             2    0  1  5/4  f    0  1  1  Q4
             2    0  1  -1/4  Q1    0  1  1  QAl
             2    0  1  5/4  Q4    0  1  1  QAl
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  QAl    0  1  1  x
             2    0  1  1/4  Q1    0  1  1  y
             2    0  1  5/4  Q4    0  1  1  y
             2    0  1  -1  x    0  1  1  y

p(afchl)   9 1    1  6  -1  f  -1  QAl  -1  y  -2  x  5/4  Q1  9/4  Q4
             2    0  1  -9/4  f    0  1  1  Q4
             2    0  1  5/4  Q1    0  1  1  QAl
             2    0  1  -9/4  Q4    0  1  1  QAl
             2    0  1  2  f    0  1  1  x
             2    0  1  1  QAl    0  1  1  x
             2    0  1  -5/4  Q1    0  1  1  y
             2    0  1  -9/4  Q4    0  1  1  y
             2    0  1  3  x    0  1  1  y

p(ames)    1 1    0  2  -1  QAl   1  y

p(daph)    9 1    0  2  1/4  Q1  5/4  Q4
             2    0  1  -5/4  f    0  1  1  Q4
             2    0  1  1/4  Q1    0  1  1  QAl
             2    0  1  -5/4  Q4    0  1  1  QAl
             2    0  1  1  f    0  1  1  x
             2    0  1  1  QAl    0  1  1  x
             2    0  1  -1/4  Q1    0  1  1  y
             2    0  1  -5/4  Q4    0  1  1  y
             2    0  1  1  x    0  1  1  y

p(ochl1)   7 1    0  2  -1  Q4   1  x
             2    0  1  1  f    0  1  1  Q4
             2    0  1  1  Q4    0  1  1  QAl
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  QAl    0  1  1  x
             2    0  1  1  Q4    0  1  1  y
             2    0  1  -1  x    0  1  1  y

p(ochl4)   8 1    0  3   1  x  -5/4  Q1  -5/4  Q4
             2    0  1  5/4  f    0  1  1  Q4
             2    0  1  -5/4  Q1    0  1  1  QAl
             2    0  1  5/4  Q4    0  1  1  QAl
             2    0  1  -1  f    0  1  1  x
             2    0  1  5/4  Q1    0  1  1  y
             2    0  1  5/4  Q4    0  1  1  y
             2    0  1  -2  x    0  1  1  y

p(f3clin)   1 1    0  1  1  f
"""

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 2021-2071.
_SF_BLOCK = """
xMgM1      5 1    1  4   1  Q1   1  QAl  -1  x  -1  y
             2    0  1  1  Q1    0  1  1  QAl
             2    0  1  -1  QAl    0  1  1  x
             2    0  1  -1  Q1    0  1  1  y
             2    0  1  1  x    0  1  1  y

xFeM1      5 1    0  2  -1  Q1   1  x
             2    0  1  -1  Q1    0  1  1  QAl
             2    0  1  1  QAl    0  1  1  x
             2    0  1  1  Q1    0  1  1  y
             2    0  1  -1  x    0  1  1  y

xAlM1      1 1    0  2  -1  QAl   1  y

xMgM23     6 1    1  3  -1/4  Q1  -1/4  Q4  -1  x
             2    0  1  1/4  f    0  1  1  Q4
             2    0  1  -1/4  Q1    0  1  1  QAl
             2    0  1  1/4  Q4    0  1  1  QAl
             2    0  1  1/4  Q1    0  1  1  y
             2    0  1  1/4  Q4    0  1  1  y

xFeM23     6 1    0  3  1/4  Q1  1/4  Q4   1  x
             2    0  1  -1/4  f    0  1  1  Q4
             2    0  1  1/4  Q1    0  1  1  QAl
             2    0  1  -1/4  Q4    0  1  1  QAl
             2    0  1  -1/4  Q1    0  1  1  y
             2    0  1  -1/4  Q4    0  1  1  y

xMgM4      7 1    1  5  -1  f   1  Q4  -1  QAl  -1  x  -1  y
             2    0  1  -1  f    0  1  1  Q4
             2    0  1  -1  Q4    0  1  1  QAl
             2    0  1  1  f    0  1  1  x
             2    0  1  1  QAl    0  1  1  x
             2    0  1  -1  Q4    0  1  1  y
             2    0  1  1  x    0  1  1  y

xFeM4      7 1    0  2  -1  Q4   1  x
             2    0  1  1  f    0  1  1  Q4
             2    0  1  1  Q4    0  1  1  QAl
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  QAl    0  1  1  x
             2    0  1  1  Q4    0  1  1  y
             2    0  1  -1  x    0  1  1  y

xFe3M4     1 1    0  1  1  f

xAlM4      1 1    0  2   1  QAl   1  y

xSiT2      1 1    1  2  -1/2  f  -1  y

xAlT2      1 1    0  2  1/2  f   1  y
"""


class Chlorite(Phase):
    abbreviation = "chl"
    sites = {
        "M1": ["Mg{2+}", "Fe{2+}", "Al{3+}"],
        "M23": ["Mg{2+}", "Fe{2+}"],
        "M4": ["Mg{2+}", "Fe{2+}", "Fe{3+}", "Al{3+}"],
        "T2": ["Si{4+}", "Al{3+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["clin", "afchl", "ames", "daph", "ochl1", "ochl4", "f3clin"]
    order_parameter_names = ("QAl", "Q1", "Q4")

    # -- petropandas Mineral metadata (from old TC_chl / TC_Chlorite, Mn-free) --
    n_oxygens = 14
    ideal_cations = None
    analytical_total_range = (85.0, 90.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 4.0, "priority": ["Si{4+}", "Al{3+}"]},
        {"name": "M", "capacity": 6.0, "priority": ["Mg{2+}", "Fe{2+}", "Al{3+}"]},
    ]

    def _raw_apfu(self, df, units="wt%"):
        """Charge-based APFU normalisation (28 positive charges)."""
        from petropandas import _calc
        from petropandas._core import _element_charge, _element_of

        cat_moles = _calc.to_apfu_by_charge(df, target_charges=28.0, units=units)
        rename = {}
        for col in cat_moles.columns:
            el = _element_of(col)
            ch = _element_charge(el)
            rename[col] = f"{el}{{{ch}+}}".replace("{1+}", "{+}")
        return cat_moles.rename(columns=rename)

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        # Not true per-site occupancies (those depend on QAl/Q1/Q4) - these are the
        # bulk cation pools `variables` needs; self.site_totals() isn't used since
        # Mg/Fe/Al are shared across sites and a naive per-site sum isn't meaningful.
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        al_t2 = 4.0 - composition["Si{4+}"]
        return pd.DataFrame(
            {
                "Fe": composition["Fe{2+}"],
                "Mg": composition["Mg{2+}"],
                "Fe3": fe3,
                "AlOct": composition["Al{3+}"] - al_t2,
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
                "y": site_fractions["AlOct"] / 2,
                "f": site_fractions["Fe3"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM1...xAlT2 (the axfile's sf block). Not used by `proportions`, but lets
        callers (and tests) check that ordered site occupancies conserve the bulk
        composition regardless of the assumed order state."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_chl = Chlorite()
