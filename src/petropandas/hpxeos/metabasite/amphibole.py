"""Clinoamphibole ('hb'), Green, White, Diener, Powell, Holland & Palin (2016), from
tc-mb51NCKFMASHTO.txt (lines 34-208). The largest/most complex model in this project:
11 end-members, 6 sites, 2 genuine order parameters (Q1, Q2) - and, as derived below,
3 more variables (z, a, k) that turn out to be just as bulk-underdetermined as Q1/Q2
even though the axfile's own legend doesn't mark them "order variable".

A(v,Na,K) mult 1, M13(Mg,Fe) mult 3, M2(Mg,Fe,Al,Fe3+,Ti) mult 2, M4(Ca,Mg,Fe,Na)
mult 2, T1(Si,Al) mult 4 (mixing), V(OH,O) mult 2 (no cation content). A fixed T2 site
holds 4 Si always (outside the mixing model; total tetrahedral cations T1+T2 = 8
always, confirmed against every end-member's formula, e.g. tr Ca2Mg5Si8O22(OH)2: A is
vacant, M13=Mg3, M2=Mg2, M4=Ca2, T1=Si4 (mixing) + T2=Si4 (fixed) = Si8 total,
2+3+2+2+8=... i.e. Ca2+Mg5+Si8=15 total cations, matching the formula exactly).

x, y, f, t, c are bulk mass-balance quantities recoverable straight from composition:
  x = Fe2+_total / (Fe2+_total + Mg_total)   (Fe, Mg only ever occupy M13/M2/M4 - all
                                               three combined, exactly matching the
                                               legend's own weighted-sum ratio, which
                                               simplifies to this because it's a ratio
                                               of the SAME weighted sum on top and
                                               bottom)
  f = Fe3+_total / 2                         (Fe3+ confined to M2, mult 2; defaults
                                               to 0 if not analyzed)
  t = Ti_total / 2                           (Ti confined to M2, mult 2)
  c = Ca_total / 2                           (Ca confined to M4, mult 2)
  y = (Al_total + Si_total - 8) / 2          (Al only ever occupies M2 or T1; AlT1 by
                                               the standard "fill T with Si first"
                                               convention over the 8 total tetrahedral
                                               cations, remainder is xAlM2 = y)

z, a, k, Q1, Q2 are NOT bulk-derivable, and z/a/k genuinely belong in the same bucket
as Q1/Q2 even though the legend doesn't call them "order variable": Na is shared
between A (mult 1) and M4 (mult 2), and unlike every other shared-cation case in this
project, there is no closure identity that pins down the split - working out
`xMgM4 + xFeM4` from the sf-block's own polynomials collapses to `1 - c - z`
(everything else cancels), which is just the M4 closure identity restated, not new
information. So M4's Mg/Fe content is *defined* in terms of z, not the other way
around - z is a genuinely free parameter here, exactly like Q1/Q2, and so are a
(A-site alkali occupancy) and k (K vs Na fraction of that alkali occupancy), which
depend on how much Na was left for A once M4 claimed its share. All five
(z, a, k, Q1, Q2) are therefore optional caller-supplied inputs via
`order_parameters`, defaulting to 0 - for z/a specifically that means "assume no Na
anywhere" (M4 is Ca+Mg+Fe only, A is fully vacant), which is the majority state among
these 11 end-members (only prgm and kprg have an occupied A site) but is a poor
assumption for genuinely sodic (Na-M4-rich) compositions like glaucophane/riebeckite -
override z (and a/k if relevant) explicitly for those.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 79-126.
_P_BLOCK = """
p(tr)      1 1    0  6  -1/2  a   1  c  -1  f  -1  t  -1  y   1  z

p(tsm)     1 1    0  4  -1/2  a   1  f   1  y  -1  z

p(prgm)    2 1    0  1   1  a
             2    0  1  -1  a    0  1  1  k

p(glm)     1 1    0  2  -1  f   1  z

p(cumm)    6 1    1  5  -1  c  -1  Q2  -1  x  -1  z  -3/2  Q1
             2    0  1  1  f    0  1  1  Q2
             2    0  1  1  Q2    0  1  1  t
             2    0  1  1  c    0  1  1  x
             2    0  1  1  Q2    0  1  1  y
             2    0  1  1  x    0  1  1  z

p(grnm)    9 1    0  3   1  x  -2  Q2  -5/2  Q1
             2    0  1  2  f    0  1  1  Q2
             2    0  1  2  Q2    0  1  1  t
             2    0  1  1  c    0  1  1  x
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  t    0  1  1  x
             2    0  1  2  Q2    0  1  1  y
             2    0  1  -1  x    0  1  1  y
             2    0  1  1  x    0  1  1  z

p(a)       6 1    0  2   1  Q2  5/2  Q1
             2    0  1  -1  f    0  1  1  Q2
             2    0  1  -1  Q2    0  1  1  t
             2    0  1  -1  c    0  1  1  x
             2    0  1  -1  Q2    0  1  1  y
             2    0  1  -1  x    0  1  1  z

p(b)       9 1    0  2  2  Q2  3/2  Q1
             2    0  1  -2  f    0  1  1  Q2
             2    0  1  -2  Q2    0  1  1  t
             2    0  1  -1  c    0  1  1  x
             2    0  1  1  f    0  1  1  x
             2    0  1  1  t    0  1  1  x
             2    0  1  -2  Q2    0  1  1  y
             2    0  1  1  x    0  1  1  y
             2    0  1  -1  x    0  1  1  z

p(mrb)     1 1    0  1  1  f

p(kprg)    1 2    0  1  1  a    0  1  1  k

p(tts)     1 1    0  1  1  t
"""

# Verbatim from tc-mb51NCKFMASHTO.txt, lines 196-256.
_SF_BLOCK = """
xvA        1 1    1  1  -1  a

xNaA       2 1    0  1   1  a
             2    0  1  -1  a    0  1  1  k

xKA        1 2    0  1  1  a    0  1  1  k

xMgM13     1 1    1  2   1  Q1  -1  x

xFeM13     1 1    0  2  -1  Q1   1  x

xMgM2      7 1    1  5  -1  f   1  Q2  -1  t  -1  x  -1  y
             2    0  1  -1  f    0  1  1  Q2
             2    0  1  -1  Q2    0  1  1  t
             2    0  1  1  f    0  1  1  x
             2    0  1  1  t    0  1  1  x
             2    0  1  -1  Q2    0  1  1  y
             2    0  1  1  x    0  1  1  y

xFeM2      7 1    0  2  -1  Q2   1  x
             2    0  1  1  f    0  1  1  Q2
             2    0  1  1  Q2    0  1  1  t
             2    0  1  -1  f    0  1  1  x
             2    0  1  -1  t    0  1  1  x
             2    0  1  1  Q2    0  1  1  y
             2    0  1  -1  x    0  1  1  y

xAlM2      1 1    0  1  1  y

xFe3M2     1 1    0  1  1  f

xTiM2      1 1    0  1  1  t

xCaM4      1 1    0  1  1  c

xMgM4      6 1    1  5  -1  c  -1  Q2  -1  x  -1  z  -3/2  Q1
             2    0  1  1  f    0  1  1  Q2
             2    0  1  1  Q2    0  1  1  t
             2    0  1  1  c    0  1  1  x
             2    0  1  1  Q2    0  1  1  y
             2    0  1  1  x    0  1  1  z

xFeM4      6 1    0  3   1  Q2   1  x  3/2  Q1
             2    0  1  -1  f    0  1  1  Q2
             2    0  1  -1  Q2    0  1  1  t
             2    0  1  -1  c    0  1  1  x
             2    0  1  -1  Q2    0  1  1  y
             2    0  1  -1  x    0  1  1  z

xNaM4      1 1    0  1  1  z

xSiT1      1 1    1  5  -1/2  f  -1/2  t  -1/2  y  1/2  z  -1/4  a

xAlT1      1 1    0  5  1/2  f  1/2  t  1/2  y  -1/2  z  1/4  a

xOHV       1 1    1  1  -1  t

xOV        1 1    0  1  1  t
"""


class Amphibole(Phase):
    abbreviation = "hb"
    sites = {
        "A": ["Na{+}", "K{+}"],
        "M13": ["Mg{2+}", "Fe{2+}"],
        "M2": ["Mg{2+}", "Fe{2+}", "Al{3+}", "Fe{3+}", "Ti{4+}"],
        "M4": ["Ca{2+}", "Mg{2+}", "Fe{2+}", "Na{+}"],
        "T": ["Si{4+}", "Al{3+}"],  # T1 mixing site only (T2 is fixed, always Si)
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = [
        "tr",
        "tsm",
        "prgm",
        "glm",
        "cumm",
        "grnm",
        "a",
        "b",
        "mrb",
        "kprg",
        "tts",
    ]
    order_parameter_names = ("z", "a", "k", "Q1", "Q2")

    # -- petropandas Mineral metadata (from existing Amp) --
    n_oxygens = 23
    ideal_cations = 15
    analytical_total_range = (96.0, 99.0)
    valence_splits = [{"element": "Fe", "method": "schumacher"}]
    site_definitions = [
        {"name": "A", "capacity": 1.0, "priority": ["K{+}", "Na{+}"]},
        {
            "name": "B",
            "capacity": 2.0,
            "priority": ["Na{+}", "Ca{2+}", "Mn{2+}", "Fe{2+}", "Mg{2+}"],
        },
        {
            "name": "C",
            "capacity": 5.0,
            "priority": [
                "Mg{2+}",
                "Fe{2+}",
                "Al{3+}",
                "Ti{4+}",
                "Cr{3+}",
                "Fe{3+}",
                "Mn{2+}",
                "Na{+}",
            ],
        },
        {"name": "T", "capacity": 8.0, "priority": ["Si{4+}", "Al{3+}"]},
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        al_t1 = 8.0 - composition["Si{4+}"]
        return pd.DataFrame(
            {
                "Fe": composition["Fe{2+}"],
                "Mg": composition["Mg{2+}"],
                "Fe3": fe3,
                "Ti": composition["Ti{4+}"],
                "Ca": composition["Ca{2+}"],
                "AlM2": composition["Al{3+}"] - al_t1,
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
                "y": site_fractions["AlM2"] / 2,
                "f": site_fractions["Fe3"] / 2,
                "t": site_fractions["Ti"] / 2,
                "c": site_fractions["Ca"] / 2,
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xvA...xOV (the axfile's sf block, including the V site's xOHV/xOV even
        though V carries no cation and isn't part of `sites`/mass balance). Not used
        by `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_hb = Amphibole()
