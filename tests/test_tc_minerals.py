"""Tests for THERMOCALC a-x solution models (petropandas.hpxeos)."""

from __future__ import annotations

import pandas as pd
import pytest

from petropandas.hpxeos.metapelite import (
    TC_bi,
    TC_cd,
    TC_chl,
    TC_ctd,
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
from petropandas.hpxeos.metabasite import TC_aug, TC_dio, TC_ol, TC_hb
from petropandas.hpxeos.igneous import TC_g_W24


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

    def test_apfu_includes_z_site_cations(self):
        """apfu() must not drop Si/Al even though hpxeos's own `sites` dict
        (used internally for the a-x mixing model) omits the always-full Z
        site -- site_definitions on the Phase subclass must be complete."""
        apfu = GARNET_DF.mineral.apfu(TC_g)
        assert "Si{4+}" in apfu.columns
        assert apfu["Si{4+}"].iloc[0] > 2.5

    def test_check_stoichiometry(self):
        result = GARNET_DF.mineral.check_stoichiometry(TC_g)
        assert (result.iloc[0] >= 0).all()
        assert (result.iloc[0] <= 1).all()


GARNET_IG_DF = pd.DataFrame(
    {
        "SiO2": [38.5],
        "Al2O3": [22.1],
        "Cr2O3": [0.0],
        "TiO2": [0.0],
        "FeO": [28.3],
        "MgO": [5.2],
        "CaO": [3.8],
    }
)


class TestTCGarnetIgneous:
    """g_W24 (Weller et al. 2024) has a different M1/M2 site model than the
    metapelite/metabasite garnet -- no Mn, but Cr/Ti-bearing end-members
    (andradite, knorringite, Ti-garnet) instead of spessartine/uvarovite."""

    def test_columns(self):
        _has_cols(
            GARNET_IG_DF.mineral.end_members(TC_g_W24),
            ["py", "alm", "gr", "andr", "knor", "tig"],
        )

    def test_sums_to_100(self):
        _sums_to_100(GARNET_IG_DF.mineral.end_members(TC_g_W24))


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
        "MnO": [0.0],
    }
)

CD_MG = pd.DataFrame(
    {
        "SiO2": [48.0],
        "Al2O3": [34.0],
        "FeO": [3.0],
        "MgO": [15.0],
        "MnO": [0.0],
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
        """MU_DF is K-dominant (K2O=10 vs Na2O=1, CaO=0.2): the K-site end-member
        family (mu/cel/fcel) should outweigh the Na/Ca family (pa/mat), even
        though Tschermak substitution (y) splits the K family further into
        celadonite components rather than leaving pure muscovite dominant."""
        r = MU_DF.mineral.end_members(TC_mu)
        k_family = r["mu"].iloc[0] + r["cel"].iloc[0] + r["fcel"].iloc[0]
        na_ca_family = r["pa"].iloc[0] + r["mat"].iloc[0]
        assert k_family > na_ca_family


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

    def test_order_parameters_accepted(self):
        """Biotite has one genuine order parameter, Q -- verify the accessor
        forwards order_parameters through to Phase.end_members/proportions."""
        default = BI_DF.mineral.end_members(TC_bi)
        ordered = BI_DF.mineral.end_members(TC_bi, order_parameters={"Q": 0.1})
        assert not default.equals(ordered)
        _sums_to_100(ordered)


# ---------------------------------------------------------------------------
# Clinopyroxene / Augite (metabasite)
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


class TestTCOmphacite:
    """TC_dio (metabasite Omphacite) is the direct successor of the old
    monolithic TC Clinopyroxene model -- same axfile abbreviation ('dio'),
    same 7 end-members."""

    def test_columns(self):
        _has_cols(
            CPX_DF.mineral.end_members(TC_dio),
            ["jd", "di", "hed", "acmm", "om", "cfm", "jac"],
        )

    def test_sums_to_100(self):
        _sums_to_100(CPX_DF.mineral.end_members(TC_dio))


class TestTCAugite:
    """Augite was split out of the old monolithic TC_dio model -- new coverage,
    no direct predecessor test to port."""

    def test_sums_to_100(self):
        _sums_to_100(CPX_DF.mineral.end_members(TC_aug))

    def test_apfu_works(self):
        apfu = CPX_DF.mineral.apfu(TC_aug)
        assert "Si{4+}" in apfu.columns


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
        """Chlorite has 3 hidden order parameters (QAl, Q1, Q4) that default to
        0 (disordered reference state) -- an absolute dominance threshold isn't
        reliable without real order-parameter input, but the Fe end-member
        (daphnite) share should still drop as the bulk gets more Mg-rich."""
        daph_fe_rich = CHL_DF.mineral.end_members(TC_chl)["daph"].iloc[0]
        daph_mg_rich = CHL_MG.mineral.end_members(TC_chl)["daph"].iloc[0]
        assert daph_mg_rich < daph_fe_rich


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
# Olivine (no predecessor in the old TC_* set)
# ---------------------------------------------------------------------------

OL_DF = pd.DataFrame({"SiO2": [40.0], "FeO": [10.0], "MgO": [49.0]})


class TestTCOlivine:
    def test_columns(self):
        _has_cols(OL_DF.mineral.end_members(TC_ol), ["fo", "fa"])

    def test_sums_to_100(self):
        _sums_to_100(OL_DF.mineral.end_members(TC_ol))

    def test_forsterite_dominant(self):
        r = OL_DF.mineral.end_members(TC_ol)
        assert r["fo"].iloc[0] > r["fa"].iloc[0]


# ---------------------------------------------------------------------------
# Amphibole (no predecessor in the old TC_* set, needs order_parameters
# for a non-degenerate, genuinely sodic composition)
# ---------------------------------------------------------------------------

AMP_DF = pd.DataFrame(
    {
        "SiO2": [52.0],
        "Al2O3": [6.0],
        "FeO": [12.0],
        "MgO": [16.0],
        "CaO": [12.0],
        "Na2O": [1.0],
        "K2O": [0.3],
        "TiO2": [0.2],
    }
)


class TestTCAmphibole:
    def test_sums_to_100(self):
        _sums_to_100(AMP_DF.mineral.end_members(TC_hb))

    def test_check_stoichiometry(self):
        result = AMP_DF.mineral.check_stoichiometry(TC_hb)
        assert (result.iloc[0] >= 0).all()


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
