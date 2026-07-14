"""Spinel ('spl_T21'), Tomlinson & Holland (2021), from tc-ig51NCKFMASHTOCr.txt. A
different, richer model generation from `hpxeos.metapelite.Spinel` (White, Powell &
Clarke 2002): real T(mult 1)/M(mult 2) crystallographic sites (not the "mixing sites,
not true sites" pseudo-grouping used there), 8 end-members incl. Cr/Ti components,
and 3 simultaneous normal/inverse order parameters - the first phase in this project
with 3.

T(Mg,Fe,Al,Fe3+) mult 1, M(Mg,Fe,Al,Fe3+,Cr,Ti) mult 2 (confirmed against every
end-member's formula, e.g. nsp "normal" MgAl2O4: T=Mg1, M=Al2, 3 total cations
matching the formula).

x, y, c, t are bulk mass-balance quantities recoverable straight from composition.
x, y are each defined by the axfile's own legend as a *properly mult-weighted*
combined ratio over T+M, which is exactly the plain bulk ratio over the same pool -
no extra split needed, unlike Garnet's coupled Mg-Ti case:
  x = (2*xFeM + xFeT) / (2*xFeM + xFeT + 2*xMgM + xMgT) = Fe2+_total / (Fe2+_total +
      Mg_total)
  y = (2*xFe3M + xFe3T) / (2*xAlM + xAlT + 2*xFe3M + xFe3T) = Fe3+_total /
      (Al_total + Fe3+_total)   (Fe3+ defaults to 0 if not analyzed, matching the
      project's usual treatment of a value that's typically calculated, not measured)
  c = Cr_total / 2   (Cr confined to M, mult 2)
  t = Ti_total        (Ti confined to M, mult 2, but `t` is *defined* as 2*xTiM, so
                        the mult-2 division cancels and t is the direct bulk total)

Q1 = -xMgM + xMgT, Q2 = -xFeM + xFeT, Q3 = -xFe3M + xFe3T describe how strongly each
divalent/trivalent cation prefers the "inverse" (T-site) position over the "normal"
(M-site) one - none is recoverable from bulk composition, so all three are optional
caller-supplied inputs defaulting to 0. Every "normal" end-member (`nsp`/`nhc`/`nmt`)
is itself a fully-ordered reference state (e.g. pure `nsp` needs Q1=1, not the
default 0) and every "inverse" end-member (`isp`/`ihc`/`imt`) needs the complementary
negative-half override (e.g. pure `isp` needs Q1=-1/2) - same "pure end-member needs
an order-parameter override" pattern as `Chlorite`'s `clin` or `Olivine`'s `cfm`.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_P_BLOCK = """
p(nsp)     2 1    1/3  4  1/3  t  -1/3  x  -1  c  2/3  Q1
             2    0  1  -1/3  t    0  1  1  x

p(isp)     2 1    2/3  3  -2/3  Q1  2/3  t  -2/3  x
             2    0  1  -2/3  t    0  1  1  x

p(nhc)      4 1    0  5  1/3  x  -1/3  y  -1  t  2/3  Q2  2/3  Q3
             2    0  1  1/3  t    0  1  1  x
             2    0  1  1/3  c    0  1  1  y
             2    0  1  1/3  t    0  1  1  y

p(ihc)     4 1    0  5  -1  t  -2/3  Q2  -2/3  Q3  2/3  x  -2/3  y
             2    0  1  2/3  t    0  1  1  x
             2    0  1  2/3  c    0  1  1  y
             2    0  1  2/3  t    0  1  1  y

p(nmt)     3 1    0  2  1/3  y  -2/3  Q3
             2    0  1  -1/3  c    0  1  1  y
             2    0  1  -1/3  t    0  1  1  y

p(imt)     3 1    0  2  2/3  Q3  2/3  y
             2    0  1  -2/3  c    0  1  1  y
             2    0  1  -2/3  t    0  1  1  y

p(picr)    1 1    0  1  1  c

p(usp)     1 1    0  1  1  t
"""

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_SF_BLOCK = """
xMgT       2 1    1/3  3  1/3  t  -1/3  x  2/3  Q1
             2    0  1  -1/3  t    0  1  1  x

xFeT       2 1    0  2  1/3  x  2/3  Q2
             2    0  1  1/3  t    0  1  1  x

xAlT       3 1    2/3  5  -1/3  t  -2/3  Q1  -2/3  Q2  -2/3  Q3  -2/3  y
             2    0  1  2/3  c    0  1  1  y
             2    0  1  2/3  t    0  1  1  y

xFe3T      3 1    0  2  2/3  Q3  2/3  y
             2    0  1  -2/3  c    0  1  1  y
             2    0  1  -2/3  t    0  1  1  y

xMgM       2 1    1/3  3  -1/3  Q1  1/3  t  -1/3  x
             2    0  1  -1/3  t    0  1  1  x

xFeM       2 1    0  2  -1/3  Q2  1/3  x
             2    0  1  1/3  t    0  1  1  x

xAlM       3 1    2/3  6  1/3  Q1  1/3  Q2  1/3  Q3  -1  c  -2/3  y  -5/6  t
             2    0  1  2/3  c    0  1  1  y
             2    0  1  2/3  t    0  1  1  y

xFe3M      3 1    0  2  -1/3  Q3  2/3  y
             2    0  1  -2/3  c    0  1  1  y
             2    0  1  -2/3  t    0  1  1  y

xCrM       1 1    0  1  1  c

xTiM       1 1    0  1  1/2  t
"""


class Spinel(Phase):
    abbreviation = "spl_T21"
    sites = {
        "T": ["Mg{2+}", "Fe{2+}", "Al{3+}", "Fe{3+}"],
        "M": ["Mg{2+}", "Fe{2+}", "Al{3+}", "Fe{3+}", "Cr{3+}", "Ti{4+}"],
    }
    optional_columns = {"Fe{3+}"}
    end_member_names = ["nsp", "isp", "nhc", "ihc", "nmt", "imt", "picr", "usp"]
    order_parameter_names = ("Q1", "Q2", "Q3")

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
                "Cr": composition["Cr{3+}"],
                "Ti": composition["Ti{4+}"],
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
                "y": site_fractions["Fe3"]
                / (site_fractions["Al"] + site_fractions["Fe3"]),
                "c": site_fractions["Cr"] / 2,
                "t": site_fractions["Ti"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgT...xTiM (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_spl_T21 = Spinel()
