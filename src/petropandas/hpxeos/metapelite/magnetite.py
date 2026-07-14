"""Magnetite ('mt1'), White, Powell, Holland & Worley (2000), from
tc-mp51MnNCKFMASHTO.txt (lines 1942-2013). "Alternative magnetite: use for
SUBSOLIDUS equilibria only!" per the axfile's own note.

T(Fe2+,Fe3+) mult 1, M(Fe2+,Fe3+,Ti) mult 2 (confirmed against the axfile's own
formula table: e.g. imt Fe3O4 = T{Fe3+=1} + M{Fe2+=1,Fe3+=1}, 3 total Fe, no other
fixed component).

x = 1 - Ti_total (Ti confined to M; this is the conventional titanomagnetite "x" in
Fe3-xTixO4). Fe2+/Fe3+ are required input columns (documenting the model) but are
not otherwise needed here: total Fe3+ turns out to equal 2x regardless of Q (the
ordering term cancels in the T+M sum), so it carries no extra information beyond x.

Q = xFe3T is O'Neil's spinel-inversion parameter: how much of the tetrahedral site's
Fe is Fe3+ versus Fe2+ (normal vs. inverse spinel) - genuinely not recoverable from
bulk composition. Unlike every other order parameter in this project, Q=0 here does
NOT mean "disordered"/no site preference (it would mean *no* Fe3+ on the tetrahedral
site at all, i.e. fully "normal" spinel ordering - the opposite of random). The
axfile's own "dmt" (disordered magnetite) end-member has Q=2/3 at x=1, matching a
truly random Fe3+/Fe2+ distribution across T and M weighted by site multiplicity:
Q_random = 2x / (2 + x). So `Q` defaults to this value (not a fixed 0) when not
supplied, keeping this phase's "no caller input" behavior actually disordered.
"""

from __future__ import annotations

import pandas as pd

from ..base import OrderParameters, Phase
from ..polynomial import evaluate_polynomials

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1974-1978.
_P_BLOCK = """
p(imt)  1 1    0  2 -2  x  3  Q

p(dmt)  1 1    0  2  3  x -3  Q

p(usp)  1 1    1  1 -1  x
"""

# Verbatim from tc-mp51MnNCKFMASHTO.txt, lines 1993-1997.
_SF_BLOCK = """
xTiM   1 1   1/2   1 -1/2  x
xFe3M  1 1    0    2   1   x -1/2 Q
xFeM   1 1   1/2   2 -1/2  x  1/2 Q
xFe3T  1 1    0    1   1   Q
xFeT   1 1    1    1  -1   Q
"""


class Magnetite(Phase):
    abbreviation = "mt1"
    sites = {
        "T": ["Fe{2+}", "Fe{3+}"],
        "M": ["Fe{2+}", "Fe{3+}", "Ti{4+}"],
    }
    end_member_names = ["imt", "dmt", "usp"]
    order_parameter_names = ("Q",)

    # -- petropandas Mineral metadata (NEW: derived from existing Spl, Fe-Ti oxide) --
    n_oxygens = 4
    ideal_cations = 3
    analytical_total_range = (93.0, 100.5)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "T", "capacity": 1.0, "priority": ["Mg{2+}", "Fe{2+}", "Mn{2+}"]},
        {
            "name": "M",
            "capacity": 2.0,
            "priority": ["Al{3+}", "Fe{3+}", "Ti{4+}", "Cr{3+}"],
        },
    ]

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"Ti": composition["Ti{4+}"]}, index=composition.index)

    def variables(
        self,
        site_fractions: pd.DataFrame,
        order_parameters: OrderParameters | None = None,
    ) -> pd.DataFrame:
        x = 1.0 - site_fractions["Ti"]
        order_parameters = order_parameters or {}
        q_raw = order_parameters.get("Q")
        if q_raw is None:
            q = 2 * x / (2 + x)
        elif isinstance(q_raw, pd.Series):
            q = q_raw
        else:
            q = pd.Series(q_raw, index=x.index)
        return pd.DataFrame({"x": x, "Q": q}, index=x.index)

    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        return evaluate_polynomials(_P_BLOCK, variables)

    def site_occupancies(self, variables: pd.DataFrame) -> pd.DataFrame:
        """xTiM, xFe3M, xFeM, xFe3T, xFeT (the axfile's sf block). Not used by
        `proportions`, but a check that _SF_BLOCK is transcribed correctly."""
        return evaluate_polynomials(_SF_BLOCK, variables)


TC_mt1 = Magnetite()
