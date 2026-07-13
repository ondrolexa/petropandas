"""Shared test fixtures for petropandas."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest


@pytest.fixture()
def diopside() -> pd.DataFrame:
    """Ideal diopside CaMgSi₂O₆ — 6 oxygens, 4 cations."""
    return pd.DataFrame(
        {"SiO2": [55.49], "MgO": [18.61], "CaO": [25.90]},
    )


@pytest.fixture()
def sanidine() -> pd.DataFrame:
    """Ideal sanidine KAlSi₃O₈ — 8 oxygens."""
    return pd.DataFrame(
        {"SiO2": [64.76], "Al2O3": [18.31], "K2O": [16.89]},
    )


@pytest.fixture()
def fe_pyroxene() -> pd.DataFrame:
    """Clinopyroxene with iron — 6 oxygens, 4 cations."""
    return pd.DataFrame(
        {
            "SiO2": [52.00],
            "Al2O3": [4.50],
            "FeO": [8.50],
            "MgO": [15.00],
            "CaO": [18.00],
            "Na2O": [1.50],
            "TiO2": [0.50],
        },
    )


@pytest.fixture()
def mn_garnet() -> pd.DataFrame:
    """Mn-bearing garnet — 12 oxygens, 8 cations."""
    return pd.DataFrame(
        {
            "SiO2": [36.50],
            "Al2O3": [20.50],
            "FeO": [25.00],
            "MnO": [15.00],
            "MgO": [1.50],
            "CaO": [1.00],
        },
    )


@pytest.fixture()
def andradite() -> pd.DataFrame:
    """Ideal andradite Ca₃Fe³⁺₂Si₃O₁₂ — 12 oxygens, 8 cations."""
    return pd.DataFrame(
        {
            "SiO2": [36.00],
            "Al2O3": [0.00],
            "FeO": [27.90],
            "CaO": [33.00],
            "MgO": [0.00],
            "MnO": [0.00],
        },
    )


@pytest.fixture()
def fe_garnet_multi() -> pd.DataFrame:
    """Multi-row garnet dataset with Fe³⁺ and Cr — 12 oxygens, 8 cations.

    Row 0: andradite-rich
    Row 1: mixed pyralspite+grossular
    Row 2: uvarovite-bearing
    """
    return pd.DataFrame(
        {
            "SiO2": [36.00, 38.00, 37.00],
            "Al2O3": [0.00, 20.00, 5.00],
            "FeO": [27.90, 20.00, 8.00],
            "Cr2O3": [0.00, 0.00, 10.00],
            "CaO": [33.00, 3.00, 28.00],
            "MgO": [0.00, 12.00, 7.00],
            "MnO": [0.00, 2.00, 0.50],
        },
    )


@pytest.fixture()
def ti_rutile() -> pd.DataFrame:
    """Ti-bearing mineral — 6 oxygens, 4 cations."""
    return pd.DataFrame(
        {
            "SiO2": [40.00],
            "Al2O3": [12.00],
            "FeO": [10.00],
            "MgO": [8.00],
            "CaO": [10.00],
            "TiO2": [18.00],
            "Na2O": [2.00],
        },
    )


@pytest.fixture()
def garnet_multi() -> pd.DataFrame:
    """Multi-row garnet dataset — 12 oxygens, 8 cations."""
    return pd.DataFrame(
        {
            "SiO2": [36.50, 38.20, 35.80],
            "Al2O3": [20.50, 22.10, 21.00],
            "FeO": [25.00, 18.50, 28.00],
            "MnO": [15.00, 2.00, 8.00],
            "MgO": [1.50, 15.00, 3.00],
            "CaO": [1.00, 4.00, 4.50],
        },
    )


@pytest.fixture()
def clinopyroxene_multi() -> pd.DataFrame:
    """Multi-row clinopyroxene dataset — 6 oxygens, 4 cations."""
    return pd.DataFrame(
        {
            "SiO2": [52.00, 48.50, 54.00],
            "Al2O3": [4.50, 8.00, 2.00],
            "FeO": [8.50, 12.00, 5.00],
            "MgO": [15.00, 10.00, 18.00],
            "CaO": [18.00, 20.00, 15.00],
            "Na2O": [1.50, 0.50, 3.00],
            "TiO2": [0.50, 1.00, 0.20],
        },
    )


@pytest.fixture()
def cr_clinopyroxene() -> pd.DataFrame:
    """Cr-Na clinopyroxene — kosmochlor-bearing, 6 oxygens, 4 cations."""
    return pd.DataFrame(
        {
            "SiO2": [50.00],
            "Al2O3": [3.00],
            "Cr2O3": [2.50],
            "FeO": [6.00],
            "MgO": [14.00],
            "CaO": [16.00],
            "Na2O": [2.00],
            "TiO2": [0.30],
        },
    )


@pytest.fixture()
def feldspar_multi() -> pd.DataFrame:
    """Multi-row feldspar dataset — 8 oxygens, 5 cations."""
    return pd.DataFrame(
        {
            "SiO2": [64.76, 52.00, 68.00],
            "Al2O3": [18.31, 30.00, 19.50],
            "CaO": [0.00, 13.00, 0.50],
            "Na2O": [1.50, 3.50, 11.00],
            "K2O": [16.89, 0.50, 0.80],
        },
    )


@pytest.fixture()
def orthopyroxene_multi() -> pd.DataFrame:
    """Multi-row orthopyroxene dataset — 6 oxygens, 4 cations."""
    return pd.DataFrame(
        {
            "SiO2": [50.00, 48.00, 52.00],
            "Al2O3": [3.00, 5.00, 1.50],
            "FeO": [15.00, 25.00, 8.00],
            "MgO": [30.00, 18.00, 35.00],
            "CaO": [1.50, 3.00, 1.00],
            "TiO2": [0.50, 1.00, 0.20],
        },
    )


@pytest.fixture()
def muscovite_multi() -> pd.DataFrame:
    """Multi-row muscovite dataset — 11 oxygens, 7.0 cations."""
    return pd.DataFrame(
        {
            "SiO2": [45.00, 42.00, 48.00],
            "Al2O3": [35.00, 30.00, 38.00],
            "FeO": [3.00, 8.00, 1.50],
            "MgO": [1.00, 3.00, 0.50],
            "TiO2": [0.50, 1.50, 0.30],
            "Na2O": [0.50, 0.30, 0.80],
            "K2O": [10.00, 9.50, 11.00],
            "BaO": [0.10, 0.20, 0.05],
        },
    )


@pytest.fixture()
def biotite() -> pd.DataFrame:
    """Ideal phlogopite K(Mg₃)(AlSi₃)O₁₀(OH)₂ — 11 oxygens, 7.0 cations."""
    return pd.DataFrame(
        {
            "SiO2": [42.70],
            "Al2O3": [11.70],
            "FeO": [0.50],
            "MgO": [29.10],
            "K2O": [10.70],
            "TiO2": [0.20],
            "MnO": [0.05],
            "Na2O": [0.10],
            "BaO": [0.00],
        },
    )


@pytest.fixture()
def biotite_multi() -> pd.DataFrame:
    """Multi-row biotite dataset — 11 oxygens, 7.0 cations.

    Row 0: near-ideal phlogopite
    Row 1: annite-rich
    Row 2: eastonite-rich (Tschermak)
    """
    return pd.DataFrame(
        {
            "SiO2": [43.00, 36.00, 36.50],
            "Al2O3": [12.50, 14.00, 17.50],
            "FeO": [3.00, 28.00, 5.00],
            "MgO": [28.00, 5.00, 18.00],
            "K2O": [10.50, 9.80, 10.20],
            "TiO2": [1.00, 3.50, 1.00],
            "MnO": [0.10, 0.50, 0.10],
            "Na2O": [0.10, 0.05, 0.10],
            "BaO": [0.00, 0.00, 0.00],
        },
    )


@pytest.fixture()
def staurolite() -> pd.DataFrame:
    """Typical Fe-Mg staurolite from metapelite — 48 oxygens."""
    return pd.DataFrame(
        {
            "SiO2": [28.0],
            "Al2O3": [53.0],
            "FeO": [13.0],
            "MgO": [2.5],
            "ZnO": [1.5],
            "MnO": [0.3],
            "TiO2": [0.7],
        },
    )


@pytest.fixture()
def staurolite_multi() -> pd.DataFrame:
    """Multi-row staurolite — 48 oxygens.

    Row 0: Fe-rich (typical)
    Row 1: Mg-rich
    Row 2: Zn-bearing
    """
    return pd.DataFrame(
        {
            "SiO2": [28.0, 29.0, 27.5],
            "Al2O3": [53.0, 55.0, 51.0],
            "FeO": [13.0, 5.0, 10.0],
            "MgO": [2.5, 8.0, 2.0],
            "ZnO": [1.5, 0.2, 7.0],
            "MnO": [0.3, 0.5, 1.0],
            "TiO2": [0.7, 0.3, 0.5],
        },
    )


@pytest.fixture()
def chlorite() -> pd.DataFrame:
    """Greenschist facies clinochlore — 28 charges."""
    return pd.DataFrame(
        {
            "SiO2": [26.0],
            "Al2O3": [21.0],
            "FeO": [20.0],
            "MgO": [18.0],
            "Cr2O3": [0.1],
            "TiO2": [0.1],
        },
    )


@pytest.fixture()
def chlorite_multi() -> pd.DataFrame:
    """Multi-row chlorite — 28 charges.

    Row 0: clinochlore-like (Mg-rich)
    Row 1: chamosite-like (Fe-rich)
    Row 2: sudoite-like (Al-rich, low Si)
    """
    return pd.DataFrame(
        {
            "SiO2": [26.0, 24.0, 30.0],
            "Al2O3": [21.0, 22.0, 28.0],
            "FeO": [8.0, 28.0, 5.0],
            "MgO": [30.0, 10.0, 20.0],
            "Cr2O3": [0.1, 0.2, 0.0],
            "TiO2": [0.1, 0.3, 0.1],
        },
    )


@pytest.fixture()
def epidote() -> pd.DataFrame:
    """Typical epidote — 12.5 oxygens, Fe³⁺."""
    return pd.DataFrame(
        {
            "SiO2": [37.5],
            "Al2O3": [23.0],
            "FeO": [12.5],
            "CaO": [22.5],
            "MnO": [0.2],
            "TiO2": [0.1],
        },
    )


@pytest.fixture()
def epidote_multi() -> pd.DataFrame:
    """Multi-row epidote — 12.5 oxygens, Fe³⁺.

    Row 0: epidote (Fe³⁺-rich)
    Row 1: clinozoisite (Al-dominant)
    Row 2: piemontite (Mn-bearing)
    """
    return pd.DataFrame(
        {
            "SiO2": [37.5, 38.5, 36.0],
            "Al2O3": [23.0, 30.0, 20.0],
            "FeO": [12.5, 2.0, 8.0],
            "CaO": [22.5, 23.5, 21.0],
            "MnO": [0.2, 0.1, 12.0],
            "TiO2": [0.1, 0.1, 0.3],
        },
    )


@pytest.fixture()
def amphibole() -> pd.DataFrame:
    """Calcic hornblende — 23 oxygens."""
    return pd.DataFrame(
        {
            "SiO2": [43.0],
            "TiO2": [1.5],
            "Al2O3": [11.0],
            "FeO": [12.0],
            "MgO": [13.0],
            "CaO": [11.5],
            "Na2O": [1.5],
            "K2O": [0.8],
            "MnO": [0.2],
            "Cr2O3": [0.1],
        },
    )


@pytest.fixture()
def amphibole_multi() -> pd.DataFrame:
    """Multi-row amphibole — 23 oxygens.

    Row 0: calcic hornblende
    Row 1: actinolite (low Al, high Mg)
    Row 2: glaucophane (Na > Ca, Mg-rich)
    """
    return pd.DataFrame(
        {
            "SiO2": [43.0, 54.0, 56.0],
            "TiO2": [1.5, 0.2, 0.1],
            "Al2O3": [11.0, 2.5, 9.0],
            "FeO": [12.0, 8.0, 10.0],
            "MgO": [13.0, 20.0, 14.0],
            "CaO": [11.5, 12.0, 2.0],
            "Na2O": [1.5, 0.5, 6.5],
            "K2O": [0.8, 0.1, 0.1],
            "MnO": [0.2, 0.1, 0.1],
            "Cr2O3": [0.1, 0.0, 0.0],
        },
    )


@pytest.fixture()
def titanite() -> pd.DataFrame:
    """Ideal titanite CaTiSiO₅ — 5 oxygens, 3 cations."""
    return pd.DataFrame(
        {
            "SiO2": [30.48],
            "TiO2": [40.83],
            "Al2O3": [0.00],
            "FeO": [0.00],
            "CaO": [28.69],
            "MnO": [0.00],
            "MgO": [0.00],
        },
    )


@pytest.fixture()
def titanite_multi() -> pd.DataFrame:
    """Multi-row titanite dataset — 5 oxygens, 3 cations.

    Row 0: ideal titanite
    Row 1: Al-bearing titanite (Al-titanite component)
    Row 2: Fe-bearing titanite (ferro-titanite component)
    """
    return pd.DataFrame(
        {
            "SiO2": [30.48, 29.50, 30.00],
            "TiO2": [40.83, 35.00, 36.00],
            "Al2O3": [0.00, 6.00, 1.50],
            "FeO": [0.00, 0.50, 4.00],
            "CaO": [28.69, 28.00, 28.50],
            "MnO": [0.00, 0.10, 0.20],
            "MgO": [0.00, 0.05, 0.10],
            "SnO2": [0.00, 0.00, 0.00],
            "Nb2O5": [0.00, 0.00, 0.00],
        },
    )


@pytest.fixture()
def titanite_sn() -> pd.DataFrame:
    """Sn-bearing titanite (malayaite component) — 5 oxygens, 3 cations."""
    return pd.DataFrame(
        {
            "SiO2": [28.00],
            "TiO2": [25.00],
            "Al2O3": [1.00],
            "FeO": [0.50],
            "CaO": [27.00],
            "SnO2": [15.00],
            "MnO": [0.00],
            "MgO": [0.00],
        },
    )


@pytest.fixture()
def chloritoid() -> pd.DataFrame:
    """Fe-Mg chloritoid from metapelite — 12 oxygens, 8 cations."""
    return pd.DataFrame(
        {
            "SiO2": [24.5],
            "Al2O3": [39.5],
            "FeO": [22.0],
            "MgO": [5.0],
            "MnO": [3.0],
            "TiO2": [0.3],
        },
    )


@pytest.fixture()
def chloritoid_multi() -> pd.DataFrame:
    """Multi-row chloritoid — 12 oxygens, 8 cations.

    Row 0: Fe-rich (Cld-dominant)
    Row 1: Mg-rich (Mgcld-dominant)
    Row 2: Mn-bearing (Mncld component)
    """
    return pd.DataFrame(
        {
            "SiO2": [24.5, 25.0, 23.0],
            "Al2O3": [39.5, 40.0, 38.0],
            "FeO": [22.0, 8.0, 18.0],
            "MgO": [5.0, 18.0, 3.0],
            "MnO": [3.0, 1.0, 14.0],
            "TiO2": [0.3, 0.5, 0.2],
        },
    )


@pytest.fixture()
def cordierite() -> pd.DataFrame:
    """Typical Mg-Fe cordierite — 18 oxygens."""
    return pd.DataFrame(
        {
            "SiO2": [48.0],
            "Al2O3": [32.5],
            "FeO": [5.0],
            "MgO": [10.5],
            "MnO": [0.3],
            "Na2O": [0.5],
            "K2O": [0.1],
            "CaO": [0.1],
        },
    )


@pytest.fixture()
def cordierite_multi() -> pd.DataFrame:
    """Multi-row cordierite — 18 oxygens.

    Row 0: Mg-rich typical cordierite
    Row 1: Fe-rich cordierite (sekaninaite-like)
    Row 2: Mn-bearing cordierite
    """
    return pd.DataFrame(
        {
            "SiO2": [48.0, 47.0, 46.0],
            "Al2O3": [32.5, 31.0, 30.0],
            "FeO": [5.0, 14.0, 6.0],
            "MgO": [10.5, 3.0, 8.0],
            "MnO": [0.3, 0.5, 7.0],
            "Na2O": [0.5, 0.3, 0.4],
            "K2O": [0.1, 0.1, 0.2],
            "CaO": [0.1, 0.1, 0.1],
        },
    )


@pytest.fixture()
def ilmenite() -> pd.DataFrame:
    """Ideal ilmenite FeTiO₃ — 3 oxygens, 2 cations."""
    return pd.DataFrame(
        {
            "SiO2": [0.00],
            "TiO2": [52.66],
            "Al2O3": [0.00],
            "FeO": [47.34],
            "MgO": [0.00],
            "MnO": [0.00],
            "Cr2O3": [0.00],
        },
    )


@pytest.fixture()
def ilmenite_multi() -> pd.DataFrame:
    """Multi-row ilmenite — 3 oxygens, 2 cations.

    Row 0: ideal ilmenite
    Row 1: Mg-bearing (geikielite component)
    Row 2: Mn-bearing (pyrophanite component)
    """
    return pd.DataFrame(
        {
            "SiO2": [0.00, 0.00, 0.00],
            "TiO2": [52.66, 50.00, 51.00],
            "Al2O3": [0.00, 0.00, 0.00],
            "FeO": [47.34, 35.00, 38.00],
            "MgO": [0.00, 14.00, 0.50],
            "MnO": [0.00, 0.50, 10.00],
            "Cr2O3": [0.00, 0.00, 0.00],
        },
    )


@pytest.fixture()
def spinel() -> pd.DataFrame:
    """Typical Mg-Al spinel — 4 oxygens, 3 cations."""
    return pd.DataFrame(
        {
            "SiO2": [0.00],
            "TiO2": [0.20],
            "Al2O3": [55.00],
            "FeO": [10.00],
            "Fe2O3": [3.00],
            "Cr2O3": [5.00],
            "MgO": [25.00],
            "MnO": [0.30],
            "ZnO": [0.50],
        },
    )


@pytest.fixture()
def spinel_multi() -> pd.DataFrame:
    """Multi-row spinel — 4 oxygens, 3 cations.

    Row 0: Mg-Al spinel (typical)
    Row 1: Fe-Al spinel (hercynite-like)
    Row 2: Cr-rich spinel (chromite-like)
    """
    return pd.DataFrame(
        {
            "SiO2": [0.00, 0.00, 0.00],
            "TiO2": [0.20, 0.50, 0.30],
            "Al2O3": [55.00, 30.00, 15.00],
            "FeO": [10.00, 25.00, 15.00],
            "Fe2O3": [3.00, 5.00, 8.00],
            "Cr2O3": [5.00, 3.00, 45.00],
            "MgO": [25.00, 5.00, 10.00],
            "MnO": [0.30, 0.20, 0.10],
            "ZnO": [0.50, 0.30, 0.10],
        },
    )


# ---------------------------------------------------------------------------
# Bulk-rock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def granite_bulk() -> pd.DataFrame:
    """Typical granite composition (wt%)."""
    return pd.DataFrame(
        {
            "SiO2": [72.0],
            "TiO2": [0.3],
            "Al2O3": [14.0],
            "Fe2O3": [1.2],
            "FeO": [1.8],
            "MnO": [0.05],
            "MgO": [0.7],
            "CaO": [1.8],
            "Na2O": [3.2],
            "K2O": [4.5],
            "P2O5": [0.12],
        },
    )


@pytest.fixture()
def basalt_bulk() -> pd.DataFrame:
    """Typical basalt composition (wt%)."""
    return pd.DataFrame(
        {
            "SiO2": [49.5],
            "TiO2": [2.0],
            "Al2O3": [14.5],
            "Fe2O3": [3.5],
            "FeO": [9.0],
            "MnO": [0.18],
            "MgO": [7.5],
            "CaO": [10.5],
            "Na2O": [2.5],
            "K2O": [0.4],
            "P2O5": [0.25],
        },
    )


@pytest.fixture()
def diorite_bulk() -> pd.DataFrame:
    """Typical diorite composition (wt%)."""
    return pd.DataFrame(
        {
            "SiO2": [59.03],
            "TiO2": [0.7],
            "Al2O3": [16.5],
            "Fe2O3": [2.5],
            "FeO": [5.0],
            "MnO": [0.12],
            "MgO": [4.0],
            "CaO": [6.5],
            "Na2O": [3.5],
            "K2O": [2.0],
            "P2O5": [0.15],
        },
    )


@pytest.fixture
def garnet_groups():
    g1 = pd.DataFrame(
        {
            "Prp": {30: 31.576, 31: 61.971},
            "Alm": {30: 46.16, 31: 25.144},
            "Sps": {30: 0.473, 31: 0.921},
            "Grs": {30: 21.792, 31: 10.96},
            "Adr": {30: 0.0, 31: 0.0},
            "Uvr": {30: 0.0, 31: 1.003},
        }
    )
    g2 = pd.DataFrame(
        {
            "Prp": {32: 75.0, 33: 8.25},
            "Alm": {32: 14.943, 33: 61.717},
            "Sps": {32: 0.682, 33: 2.218},
            "Grs": {32: 9.05, 33: 27.632},
            "Adr": {32: 0.0, 33: 0.0},
            "Uvr": {32: 0.325, 33: 0.184},
        }
    )
    g3 = pd.DataFrame(
        {
            "Prp": {
                34: 69.902,
                35: 56.533,
                36: 18.767,
                37: 26.87,
                38: 22.124,
                39: 35.517,
            },
            "Alm": {
                34: 15.523,
                35: 31.238,
                36: 56.413,
                37: 49.652,
                38: 53.149,
                39: 44.891,
            },
            "Sps": {34: 0.961, 35: 0.756, 36: 1.576, 37: 0.61, 38: 1.034, 39: 0.96},
            "Grs": {
                34: 3.134,
                35: 11.179,
                36: 23.244,
                37: 22.868,
                38: 23.693,
                39: 15.808,
            },
            "Adr": {34: 4.743, 35: 0.0, 36: 0.0, 37: 0.0, 38: 0.0, 39: 2.748},
            "Uvr": {34: 5.737, 35: 0.294, 36: 0.0, 37: 0.0, 38: 0.0, 39: 0.076},
        }
    )
    return g1, g2, g3


@pytest.fixture
def profile_groups():
    """Two profile DataFrames for ProfilePlot tests."""
    p1 = pd.DataFrame(
        {
            "CaO": [14.661, 7.5, 3.242, 3.111, 3.714],
            "FeO": [78.417, 84.421, 87.682, 87.413, 86.992],
            "MgO": [4.436, 6.106, 6.986, 7.299, 7.224],
            "MnO": [2.454, 1.935, 1.894, 1.885, 1.756],
        },
        index=pd.RangeIndex(start=0, stop=5, name="Point"),
    )
    p2 = pd.DataFrame(
        {
            "ZnO": [0.5, 0.3, 0.1, 0.2, 0.4],
            "Na2O": [1.2, 1.5, 1.8, 1.6, 1.3],
        },
        index=pd.RangeIndex(start=0, stop=5, name="Point"),
    )
    return p1, p2


@pytest.fixture
def garnet_profile():
    gp = pd.DataFrame(
        {
            "Prp": {
                807: 4.436,
                808: 6.106,
                809: 6.986,
                810: 7.299,
                811: 7.224,
                812: 7.29,
                813: 6.923,
                814: 7.136,
                815: 6.485,
                816: 6.454,
                817: 6.497,
                818: 6.352,
                819: 5.873,
                820: 5.044,
                821: 4.156,
            },
            "Alm": {
                807: 78.417,
                808: 84.421,
                809: 87.682,
                810: 87.413,
                811: 86.992,
                812: 86.759,
                813: 85.3,
                814: 85.912,
                815: 84.325,
                816: 83.018,
                817: 84.442,
                818: 84.765,
                819: 82.832,
                820: 80.767,
                821: 77.814,
            },
            "Sps": {
                807: 2.454,
                808: 1.935,
                809: 1.894,
                810: 1.885,
                811: 1.756,
                812: 1.921,
                813: 1.663,
                814: 1.821,
                815: 1.839,
                816: 1.833,
                817: 1.864,
                818: 1.879,
                819: 1.916,
                820: 2.098,
                821: 2.31,
            },
            "Grs": {
                807: 14.661,
                808: 7.5,
                809: 3.242,
                810: 3.111,
                811: 3.714,
                812: 3.857,
                813: 5.814,
                814: 5.083,
                815: 7.309,
                816: 8.646,
                817: 7.154,
                818: 6.991,
                819: 9.367,
                820: 12.091,
                821: 15.688,
            },
            "Adr": {
                807: 0.0,
                808: 0.0,
                809: 0.152,
                810: 0.291,
                811: 0.308,
                812: 0.124,
                813: 0.3,
                814: 0.0,
                815: 0.0,
                816: 0.0,
                817: 0.0,
                818: 0.0,
                819: 0.0,
                820: 0.0,
                821: 0.0,
            },
            "Uvr": {
                807: 0.032,
                808: 0.039,
                809: 0.045,
                810: 0.0,
                811: 0.006,
                812: 0.048,
                813: 0.0,
                814: 0.049,
                815: 0.042,
                816: 0.048,
                817: 0.042,
                818: 0.013,
                819: 0.013,
                820: 0.0,
                821: 0.033,
            },
        }
    )
    return gp
