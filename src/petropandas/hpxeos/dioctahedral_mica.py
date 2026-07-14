"""Shared site model for the K-white-mica family: muscovite ('mu') and margarite ('ma')
in the metapelite axfile (tc-mp51MnNCKFMASHTO.txt), and muscovite ('mu') in the
metabasite axfile (tc-mb51NCKFMASHTO.txt), all use the identical A(K,Na,Ca)
M2A(Mg,Fe,Al) M2B(Al,Fe3+) T1_2(Si,Al) site model and x,y,f,n,c parameterization (down
to identical p(...)/sf polynomials, differing only in end-member names - margarite is
simply the Ca-dominant relabeling of the same mica model muscovite uses, and
metabasite's muscovite differs from metapelite's only in naming its Ca end-member "mam"
instead of "mat"). This base class lives at the hpxeos top level (not inside any one
axfile set's subpackage) because it's shared across sets, same reasoning as `Phase`
itself; it factors out the shared mass balance, and each subclass only supplies its own
`_P_BLOCK`, `_SF_BLOCK` and `end_member_names`.

A/M2A/M2B mult 1, T1 mult 2, plus a fixed T2 site holding 2 Si always (outside the
mixing model, total tetrahedral cations T1+T2 = 4 always - the standard "AlIV = 4 -
Si" mica convention). Ba, Cr, Mn, Ti present in real analyses have no end-member in
this model and are excluded from site allocation, matching the project's earlier
garnet/chlorite convention for unmodeled trace cations.

No hidden order-disorder parameter: x, y, f, n, c are all directly recoverable from
bulk composition -
  x = Fe2+_total / (Fe2+_total + Mg_total)         (Fe, Mg only ever occupy M2A)
  n = Na_total / (K_total + Na_total + Ca_total)   (K, Na, Ca only ever occupy A)
  c = Ca_total / (K_total + Na_total + Ca_total)
  f = Fe3+_total                                   (Fe3+ confined to M2B, mult 1;
                                                     defaults to 0 if not analyzed)
  y = xAlM2A, recovered from the T1 Al mass balance: AlT1 = 4 - Si_total (tetrahedral
      Al by difference), and since xAlT1 = (c + y)/2 (T1 mult 2), y = AlT1 - c.
"""

from __future__ import annotations

import pandas as pd

from .base import Phase


class DioctahedralMica(Phase):
    sites = {
        "A": ["K{+}", "Na{+}", "Ca{2+}"],
        "M2A": ["Mg{2+}", "Fe{2+}", "Al{3+}"],
        "M2B": ["Al{3+}", "Fe{3+}"],
        "T1": ["Si{4+}", "Al{3+}"],
    }
    optional_columns = {"Fe{3+}"}

    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        if "Fe{3+}" in composition:
            fe3 = composition["Fe{3+}"]
        else:
            fe3 = pd.Series(0.0, index=composition.index)
        return pd.DataFrame(
            {
                "Fe": composition["Fe{2+}"],
                "Mg": composition["Mg{2+}"],
                "Na": composition["Na{+}"],
                "Ca": composition["Ca{2+}"],
                "ATotal": composition["K{+}"]
                + composition["Na{+}"]
                + composition["Ca{2+}"],
                "AlT1": 4.0 - composition["Si{4+}"],
                "Fe3": fe3,
            },
            index=composition.index,
        )

    def variables(
        self, site_fractions: pd.DataFrame, order_parameters=None
    ) -> pd.DataFrame:
        c = site_fractions["Ca"] / site_fractions["ATotal"]
        n = site_fractions["Na"] / site_fractions["ATotal"]
        return pd.DataFrame(
            {
                "x": site_fractions["Fe"]
                / (site_fractions["Fe"] + site_fractions["Mg"]),
                "y": site_fractions["AlT1"] - c,
                "f": site_fractions["Fe3"],
                "n": n,
                "c": c,
            },
            index=site_fractions.index,
        )
