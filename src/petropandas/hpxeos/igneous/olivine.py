"""Olivine ('ol_H18'), Holland, Green & Powell (2018), from tc-ig51NCKFMASHTOCr.txt.
A richer CFMS model than `hpxeos.metabasite.Olivine` (Holland & Powell 2011, Mg/Fe
only, no Ca, no order parameter) - this one adds monticellite (Ca-olivine) and an
Fe/Mg ordering variable between the two sites.

M1(Mg,Fe) M2(Mg,Fe,Ca), both mult 1 (confirmed against every end-member's formula,
e.g. mont CaMgSiO4: M1=Mg1, M2=Ca1, 2 total octahedral cations matching the formula).

x, c are bulk mass-balance quantities recoverable straight from composition:
  x = Fe2+_total / (Fe2+_total + Mg_total)   (Fe, Mg only ever occupy M1/M2; this is
                                               a combined ratio over both sites, which
                                               is exactly what the axfile's own
                                               (xFeM1+xFeM2)/(...) definition reduces
                                               to since M1, M2 are both mult 1)
  c = Ca_total                               (Ca confined to M2, mult 1)

Q = x - xFeM1/(xFeM1+xMgM1) describes how strongly Fe/Mg order between M1 and M2
relative to the bulk ratio - not recoverable from bulk composition, so it is an
optional caller-supplied input defaulting to 0 (disordered). Note the pure `cfm`
end-member (Mg entirely on M1, Fe entirely on M2) is itself the *fully ordered*
state at x=0.5, i.e. Q=0.5, not Q=0 - the axfile's own reference "check" row for cfm
confirms this (same "pure end-member needs a nonzero order-parameter override" case
as `Chlorite`'s pure `clin`).
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_P_BLOCK = """
p(mont)    1 1    0  1  1  c

p(fa)      1 1    0  2  -1  Q   1  x

p(fo)      2 1    1  3  -1  c  -1  Q  -1  x
             2    0  1  1  c    0  1  1  x

p(cfm)     2 1    0  1  2  Q
             2    0  1  -1  c    0  1  1  x
"""

# Verbatim from tc-ig51NCKFMASHTOCr.txt.
_SF_BLOCK = """
xMgM1      1 1    1  2   1  Q  -1  x

xFeM1      1 1    0  2  -1  Q   1  x

xMgM2      2 1    1  3  -1  c  -1  Q  -1  x
             2    0  1  1  c    0  1  1  x

xFeM2      2 1    0  2   1  Q   1  x
             2    0  1  -1  c    0  1  1  x

xCaM2      1 1    0  1  1  c
"""


class Olivine(Phase):
    abbreviation = "ol_H18"
    sites = {
        "M1": ["Mg{2+}", "Fe{2+}"],
        "M2": ["Mg{2+}", "Fe{2+}", "Ca{2+}"],
    }
    end_member_names = ["mont", "fa", "fo", "cfm"]
    order_parameter_names = ("Q",)

    # -- petropandas Mineral metadata (NEW: standard (Mg,Fe)2SiO4 olivine formula) --
    n_oxygens = 4
    ideal_cations = 3
    analytical_total_range = (98.0, 101.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 1.0, "priority": ["Si{4+}"]},
        {
            "name": "M",
            "capacity": 2.0,
            "priority": ["Mg{2+}", "Fe{2+}", "Mn{2+}", "Ca{2+}"],
        },
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Fe": composition["Fe{2+}"],
                "Mg": composition["Mg{2+}"],
                "Ca": composition["Ca{2+}"],
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
                "c": site_fractions["Ca"],
                **order,
            },
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xMgM1...xCaM2 (the axfile's sf block). Not used by `proportions`, but a
        check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_ol_H18 = Olivine()
