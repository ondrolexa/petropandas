"""Tests for THERMOCALC-compatible mineral end-member calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from petropandas._minerals import (
    TC_bi,
    TC_cd,
    TC_chl,
    TC_ctd,
    TC_dio,
    TC_ep,
    TC_g,
    TC_ilmm,
    TC_k4tr,
    TC_ma,
    TC_mu,
    TC_opx,
    TC_pl4tr,
    TC_sa,
    TC_sp,
    TC_st,
)


def _sums_to_100(result: pd.DataFrame, tol: float = 0.01) -> None:
    totals = result.sum(axis=1)
    assert ((totals - 100).abs() < tol).all(), f"Row sums: {totals.tolist()}"


def _has_cols(result: pd.DataFrame, expected: list[str]) -> None:
    assert list(result.columns) == expected


# ---------------------------------------------------------------------------
# Garnet
# ---------------------------------------------------------------------------

GARNET_DF = pd.DataFrame(
    {
        "SiO2": [38.5],
        "Al2O3": [22.1],
        "FeO": [28.3],
        "MgO": [5.2],
        "CaO": [3.8],
        "MnO": [1.5],
    }
)

GARNET_FE = pd.DataFrame(
    {
        "SiO2": [38.0],
        "Al2O3": [21.5],
        "FeO": [32.0],
        "MgO": [3.0],
        "CaO": [4.0],
        "MnO": [1.0],
    }
)

GARNET_MG = pd.DataFrame(
    {
        "SiO2": [42.0],
        "Al2O3": [23.0],
        "FeO": [10.0],
        "MgO": [20.0],
        "CaO": [4.0],
        "MnO": [0.5],
    }
)


class TestTCGarnet:
    def test_columns(self):
        _has_cols(
            GARNET_DF.mineral.end_members(TC_g), ["py", "alm", "spss", "gr", "kho"]
        )

    def test_sums_to_100(self):
        _sums_to_100(GARNET_DF.mineral.end_members(TC_g))

    def test_fe_rich(self):
        r = GARNET_FE.mineral.end_members(TC_g)
        assert r["alm"].iloc[0] > r["py"].iloc[0]

    def test_mg_rich(self):
        r = GARNET_MG.mineral.end_members(TC_g)
        assert r["py"].iloc[0] > r["alm"].iloc[0]


# ---------------------------------------------------------------------------
# Feldspar (pl4tr)
# ---------------------------------------------------------------------------

FELD_DF = pd.DataFrame(
    {
        "SiO2": [60.0],
        "Al2O3": [25.0],
        "CaO": [7.0],
        "Na2O": [7.0],
        "K2O": [1.0],
    }
)

FELD_PLAG = pd.DataFrame(
    {
        "SiO2": [55.0],
        "Al2O3": [28.0],
        "CaO": [12.0],
        "Na2O": [4.0],
        "K2O": [0.5],
    }
)


class TestTCFeldsparPl4tr:
    def test_columns(self):
        _has_cols(FELD_DF.mineral.end_members(TC_pl4tr), ["ab", "an", "san"])

    def test_sums_to_100(self):
        _sums_to_100(FELD_DF.mineral.end_members(TC_pl4tr))

    def test_plagioclase_rich(self):
        r = FELD_PLAG.mineral.end_members(TC_pl4tr)
        assert r["an"].iloc[0] > r["ab"].iloc[0]


class TestTCFeldsparK4tr:
    def test_columns(self):
        _has_cols(FELD_DF.mineral.end_members(TC_k4tr), ["ab", "an", "san"])

    def test_sums_to_100(self):
        _sums_to_100(FELD_DF.mineral.end_members(TC_k4tr))

    def test_pl4tr_k4tr_agree(self):
        r1 = FELD_DF.mineral.end_members(TC_pl4tr)
        r2 = FELD_DF.mineral.end_members(TC_k4tr)
        for col in r1.columns:
            assert r1[col].iloc[0] == pytest.approx(r2[col].iloc[0], abs=0.1)


# ---------------------------------------------------------------------------
# Epidote
# ---------------------------------------------------------------------------

EP_DF = pd.DataFrame(
    {
        "SiO2": [38.0],
        "Al2O3": [25.0],
        "FeO": [10.0],
        "CaO": [23.0],
    }
)


class TestTCEpidote:
    def test_columns(self):
        _has_cols(EP_DF.mineral.end_members(TC_ep), ["cz", "ep", "fep"])

    def test_sums_to_100(self):
        _sums_to_100(EP_DF.mineral.end_members(TC_ep))

    def test_all_fe3(self):
        r = EP_DF.mineral.end_members(TC_ep)
        assert r["fep"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Cordierite
# ---------------------------------------------------------------------------

CD_DF = pd.DataFrame(
    {
        "SiO2": [48.0],
        "Al2O3": [34.0],
        "FeO": [10.0],
        "MgO": [8.0],
    }
)

CD_MG = pd.DataFrame(
    {
        "SiO2": [48.0],
        "Al2O3": [34.0],
        "FeO": [3.0],
        "MgO": [15.0],
    }
)


class TestTCCordierite:
    def test_columns(self):
        _has_cols(CD_DF.mineral.end_members(TC_cd), ["crd", "fcrd", "hcrd", "mncd"])

    def test_sums_to_100(self):
        _sums_to_100(CD_DF.mineral.end_members(TC_cd))

    def test_mg_rich(self):
        r = CD_MG.mineral.end_members(TC_cd)
        assert r["crd"].iloc[0] > r["fcrd"].iloc[0]


# ---------------------------------------------------------------------------
# Chloritoid
# ---------------------------------------------------------------------------

CTD_DF = pd.DataFrame(
    {
        "SiO2": [25.0],
        "Al2O3": [40.0],
        "FeO": [22.0],
        "MgO": [5.0],
        "MnO": [0.5],
    }
)


class TestTCChloritoid:
    def test_columns(self):
        _has_cols(CTD_DF.mineral.end_members(TC_ctd), ["mctd", "fctd", "mnct", "ctdo"])

    def test_sums_to_100(self):
        _sums_to_100(CTD_DF.mineral.end_members(TC_ctd))


# ---------------------------------------------------------------------------
# Spinel
# ---------------------------------------------------------------------------

SP_DF = pd.DataFrame(
    {
        "SiO2": [0.5],
        "Al2O3": [55.0],
        "FeO": [35.0],
        "MgO": [5.0],
        "Fe2O3": [3.0],
        "TiO2": [1.5],
    }
)

SP_AL = pd.DataFrame(
    {
        "SiO2": [0.5],
        "Al2O3": [60.0],
        "FeO": [30.0],
        "MgO": [8.0],
        "Fe2O3": [0.5],
        "TiO2": [0.1],
    }
)


class TestTCSpinel:
    def test_columns(self):
        _has_cols(SP_DF.mineral.end_members(TC_sp), ["herc", "sp", "mt", "usp"])

    def test_sums_to_100(self):
        _sums_to_100(SP_DF.mineral.end_members(TC_sp))

    def test_al_rich(self):
        r = SP_AL.mineral.end_members(TC_sp)
        assert r["sp"].iloc[0] + r["herc"].iloc[0] > 90


# ---------------------------------------------------------------------------
# Margarite
# ---------------------------------------------------------------------------

MA_DF = pd.DataFrame(
    {
        "SiO2": [30.0],
        "Al2O3": [50.0],
        "FeO": [2.0],
        "MgO": [0.5],
        "CaO": [12.0],
        "Na2O": [0.5],
        "K2O": [0.1],
    }
)


class TestTCMargarite:
    def test_columns(self):
        _has_cols(
            MA_DF.mineral.end_members(TC_ma),
            ["mut", "celt", "fcelt", "pat", "ma", "fmu"],
        )

    def test_sums_to_100(self):
        _sums_to_100(MA_DF.mineral.end_members(TC_ma))


# ---------------------------------------------------------------------------
# Muscovite
# ---------------------------------------------------------------------------

MU_DF = pd.DataFrame(
    {
        "SiO2": [45.0],
        "Al2O3": [35.0],
        "FeO": [3.0],
        "MgO": [1.0],
        "CaO": [0.2],
        "Na2O": [1.0],
        "K2O": [10.0],
    }
)


class TestTCMuscovite:
    def test_columns(self):
        _has_cols(
            MU_DF.mineral.end_members(TC_mu), ["mu", "cel", "fcel", "pa", "mat", "fmu"]
        )

    def test_sums_to_100(self):
        _sums_to_100(MU_DF.mineral.end_members(TC_mu))

    def test_k_rich(self):
        r = MU_DF.mineral.end_members(TC_mu)
        assert r["mu"].iloc[0] > 50


# ---------------------------------------------------------------------------
# Biotite
# ---------------------------------------------------------------------------

BI_DF = pd.DataFrame(
    {
        "SiO2": [35.0],
        "Al2O3": [18.0],
        "FeO": [22.0],
        "MgO": [10.0],
        "TiO2": [3.0],
        "MnO": [0.3],
        "K2O": [9.0],
        "Na2O": [0.3],
    }
)


class TestTCBiotite:
    def test_columns(self):
        _has_cols(
            BI_DF.mineral.end_members(TC_bi),
            ["phl", "annm", "obi", "east", "tbi", "fbi", "mmbi"],
        )

    def test_sums_to_100(self):
        _sums_to_100(BI_DF.mineral.end_members(TC_bi))


# ---------------------------------------------------------------------------
# Clinopyroxene
# ---------------------------------------------------------------------------

CPX_DF = pd.DataFrame(
    {
        "SiO2": [50.0],
        "Al2O3": [5.0],
        "FeO": [8.0],
        "MgO": [12.0],
        "CaO": [20.0],
        "Na2O": [3.0],
        "TiO2": [0.5],
    }
)


class TestTCClinopyroxene:
    def test_columns(self):
        _has_cols(
            CPX_DF.mineral.end_members(TC_dio),
            ["jd", "di", "hed", "acmm", "om", "cfm", "jac"],
        )

    def test_sums_to_100(self):
        _sums_to_100(CPX_DF.mineral.end_members(TC_dio))


# ---------------------------------------------------------------------------
# Orthopyroxene
# ---------------------------------------------------------------------------

OPX_DF = pd.DataFrame(
    {
        "SiO2": [50.0],
        "Al2O3": [3.0],
        "FeO": [20.0],
        "MgO": [24.0],
        "CaO": [1.0],
        "TiO2": [0.2],
        "MnO": [0.5],
    }
)


class TestTCOrthopyroxene:
    def test_columns(self):
        _has_cols(
            OPX_DF.mineral.end_members(TC_opx),
            ["en", "fs", "fm", "mgts", "fopx", "mnopx", "odi"],
        )

    def test_sums_to_100(self):
        _sums_to_100(OPX_DF.mineral.end_members(TC_opx))


# ---------------------------------------------------------------------------
# Sapphirine
# ---------------------------------------------------------------------------

SA_DF = pd.DataFrame(
    {
        "SiO2": [15.0],
        "Al2O3": [40.0],
        "FeO": [15.0],
        "MgO": [28.0],
    }
)


class TestTCSapphirine:
    def test_columns(self):
        _has_cols(
            SA_DF.mineral.end_members(TC_sa), ["spr4", "spr5", "fspm", "spro", "ospr"]
        )

    def test_sums_to_100(self):
        _sums_to_100(SA_DF.mineral.end_members(TC_sa))


# ---------------------------------------------------------------------------
# Staurolite
# ---------------------------------------------------------------------------

ST_DF = pd.DataFrame(
    {
        "SiO2": [28.0],
        "Al2O3": [53.0],
        "FeO": [13.0],
        "MgO": [2.0],
        "MnO": [0.3],
        "TiO2": [0.8],
    }
)


class TestTCStaurolite:
    def test_columns(self):
        _has_cols(
            ST_DF.mineral.end_members(TC_st), ["mstm", "fst", "mnstm", "msto", "mstt"]
        )

    def test_sums_to_100(self):
        _sums_to_100(ST_DF.mineral.end_members(TC_st))


# ---------------------------------------------------------------------------
# Chlorite
# ---------------------------------------------------------------------------

CHL_DF = pd.DataFrame(
    {
        "SiO2": [25.0],
        "Al2O3": [22.0],
        "FeO": [18.0],
        "MgO": [18.0],
        "MnO": [0.3],
    }
)

CHL_MG = pd.DataFrame(
    {
        "SiO2": [30.0],
        "Al2O3": [20.0],
        "FeO": [5.0],
        "MgO": [30.0],
        "MnO": [0.1],
    }
)


class TestTCChlorite:
    def test_columns(self):
        _has_cols(
            CHL_DF.mineral.end_members(TC_chl),
            ["clin", "afchl", "ames", "daph", "ochl1", "ochl4", "f3clin", "mmchl"],
        )

    def test_sums_to_100(self):
        _sums_to_100(CHL_DF.mineral.end_members(TC_chl))

    def test_mg_rich(self):
        r = CHL_MG.mineral.end_members(TC_chl)
        assert r["clin"].iloc[0] + r["afchl"].iloc[0] > 50


# ---------------------------------------------------------------------------
# Ilmenite
# ---------------------------------------------------------------------------

ILM_DF = pd.DataFrame(
    {
        "SiO2": [0.5],
        "Al2O3": [0.5],
        "FeO": [40.0],
        "MgO": [2.0],
        "MnO": [2.0],
        "TiO2": [50.0],
        "Fe2O3": [3.0],
    }
)


class TestTCIlmenite:
    def test_columns(self):
        _has_cols(
            ILM_DF.mineral.end_members(TC_ilmm), ["oilm", "dilm", "dhem", "geik", "pnt"]
        )

    def test_sums_to_100(self):
        _sums_to_100(ILM_DF.mineral.end_members(TC_ilmm))


# ---------------------------------------------------------------------------
# Multi-row support
# ---------------------------------------------------------------------------


class TestMultiRow:
    def test_multi_row_garnet(self):
        df = pd.DataFrame(
            {
                "SiO2": [38.5, 42.0],
                "Al2O3": [22.1, 23.0],
                "FeO": [28.3, 10.0],
                "MgO": [5.2, 20.0],
                "CaO": [3.8, 4.0],
                "MnO": [1.5, 0.5],
            }
        )
        result = df.mineral.end_members(TC_g)
        assert len(result) == 2
        _sums_to_100(result)
        assert result["alm"].iloc[0] > result["alm"].iloc[1]
        assert result["py"].iloc[0] < result["py"].iloc[1]

    def test_multi_row_feldspar(self):
        df = pd.DataFrame(
            {
                "SiO2": [60.0, 55.0],
                "Al2O3": [25.0, 28.0],
                "CaO": [7.0, 12.0],
                "Na2O": [7.0, 4.0],
                "K2O": [1.0, 0.5],
            }
        )
        result = df.mineral.end_members(TC_pl4tr)
        assert len(result) == 2
        _sums_to_100(result)
