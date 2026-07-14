"""Epidote ('ep'), Holland & Powell (2011), from tc-mp51MnNCKFMASHTO.txt (lines 385-427).

M1(Al,Fe3+) M3(Al,Fe3+), each mult 1. A fixed M2 site (always Al) also exists in the
real structure but is outside the mixing model, so is never referenced below.

f = (xFe3M1+xFe3M3)/2 is Fe3+_total/2 directly, since Fe3+ never occupies M2 and each
of M1, M3 has mult 1 (Al{3+} carries no extra information for this reverse calculation:
it's exactly the complement of Fe3+ on M1+M3, so isn't needed to derive `f`, though it
is still required as an input column since a real analysis should report it).

Q = f - xFe3M1 describes how strongly Fe3+ prefers M3 over M1 ("ep" is, by definition,
the fully-ordered end-member: Fe3+ entirely on M3). This ordering state is not
recoverable from bulk composition, so - per the project's convention for
order-disorder phases - it is an optional caller-supplied input defaulting to 0
(fully disordered). Note natural epidote is typically strongly ordered (Q close to
f), so the Q=0 default is not a physically typical assumption here.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase, resolve_order_parameters
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 414-417.
_P_BLOCK = """
p(cz)    1  1    1  2 -1  f  -1  Q
p(ep)    1  1    0  1  2  Q
p(fep)   1  1    0  2  1  f  -1  Q
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 425-428.
_SF_BLOCK = """
xFeM1   1  1    0  2  1  f  -1  Q
xAlM1   1  1    1  2 -1  f   1  Q
xFeM3   1  1    0  2  1  f   1  Q
xAlM3   1  1    1  2 -1  f  -1  Q
"""


class Epidote(Phase):
    abbreviation = "ep"
    sites = {
        "M1": ["Al{3+}", "Fe{3+}"],
        "M3": ["Al{3+}", "Fe{3+}"],
    }
    end_member_names = ["cz", "ep", "fep"]
    order_parameter_names = ("Q",)

    # -- petropandas Mineral metadata (from old TC_ep / TC_Epidote) --
    n_oxygens = 12.5
    ideal_cations = None
    analytical_total_range = (98.0, 102.0)
    valence_splits = []
    site_definitions = [
        {"name": "M1", "capacity": 1.0, "priority": ["Al{3+}", "Fe{3+}"]},
        {"name": "M3", "capacity": 1.0, "priority": ["Al{3+}", "Fe{3+}"]},
    ]

    def _preprocess_oxides(self, oxide_df):
        """All Fe as Fe3+ (convert FeO -> Fe2O3 before normalisation)."""
        from petropandas import _calc

        return _calc.feo_to_fe2o3(oxide_df)

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"Fe3": composition["Fe{3+}"]}, index=composition.index)

    def variables(
        self,
        site_fractions: pd.DataFrame,
        order_parameters: OrderParameters | None = None,
    ) -> pd.DataFrame:
        order = resolve_order_parameters(
            order_parameters, self.order_parameter_names, site_fractions.index
        )
        return pd.DataFrame(
            {"f": site_fractions["Fe3"] / 2, **order},
            index=site_fractions.index,
        )

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xFeM1, xAlM1, xFeM3, xAlM3 (the axfile's sf block). Not used by
        `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_ep = Epidote()
