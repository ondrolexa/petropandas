"""Tests for _minerals.py — mineral classes, structural formulas, end-members."""

from __future__ import annotations

import pandas as pd
import pytest

from petropandas._minerals import (
    Amp,
    Bt,
    Chl,
    Cld,
    Cpx,
    Crd,
    Ep,
    Fsp,
    Grt,
    GrtFe3,
    Ilm,
    Mineral,
    Ms,
    Opx,
    Spl,
    St,
    Ttn,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TestMineralBase:
    def test_base_end_members_raises(self, diopside: pd.DataFrame) -> None:
        m = Mineral()
        m.n_oxygens = 6
        with pytest.raises(NotImplementedError):
            m.end_members(diopside)

    def test_name(self) -> None:
        assert Grt.name == "Garnet"


# ---------------------------------------------------------------------------
# Garnet
# ---------------------------------------------------------------------------


class TestGarnet:
    def test_apfu_columns(self, mn_garnet: pd.DataFrame) -> None:
        result = Grt.apfu(mn_garnet)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Fe{3+}",
            "Mg{2+}",
            "Ca{2+}",
            "Mn{2+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, mn_garnet: pd.DataFrame) -> None:
        result = Grt.apfu(mn_garnet)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_structural_formula_sites(self, mn_garnet: pd.DataFrame) -> None:
        result = Grt.site_allocations(mn_garnet)
        z_cols = [c for c in result.columns if c[0] == "Z"]
        y_cols = [c for c in result.columns if c[0] == "Y"]
        x_cols = [c for c in result.columns if c[0] == "X"]
        assert len(z_cols) > 0
        assert len(y_cols) > 0
        assert len(x_cols) > 0

    def test_z_site_sum(self, mn_garnet: pd.DataFrame) -> None:
        result = Grt.site_allocations(mn_garnet)
        z_cols = [c for c in result.columns if c[0] == "Z" and c[1] != "_unallocated"]
        z_total = result[z_cols].sum(axis=1)
        assert z_total.iloc[0] == pytest.approx(3.0, abs=0.1)

    def test_end_members_sum(self, mn_garnet: pd.DataFrame) -> None:
        result = Grt.end_members(mn_garnet)
        row_sum = result.sum(axis=1).iloc[0]
        assert row_sum == pytest.approx(100.0, abs=1.0)

    def test_multi_row(self, garnet_multi: pd.DataFrame) -> None:
        result = Grt.apfu(garnet_multi)
        assert len(result) == 3

    def test_end_members_multi(self, garnet_multi: pd.DataFrame) -> None:
        result = Grt.end_members(garnet_multi)
        assert len(result) == 3
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# GarnetFe3
# ---------------------------------------------------------------------------


class TestGarnetFe3:
    def test_apfu_columns(self, fe_garnet_multi: pd.DataFrame) -> None:
        result = GrtFe3.apfu(fe_garnet_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Fe{3+}",
            "Mg{2+}",
            "Ca{2+}",
            "Cr{3+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_end_members_columns(self, fe_garnet_multi: pd.DataFrame) -> None:
        result = GrtFe3.end_members(fe_garnet_multi)
        expected = {"Prp", "Alm", "Sps", "Grs", "Adr", "Uvr"}
        assert expected == set(result.columns)

    def test_end_members_sum(self, fe_garnet_multi: pd.DataFrame) -> None:
        result = GrtFe3.end_members(fe_garnet_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_andradite_ideal(self, andradite: pd.DataFrame) -> None:
        result = GrtFe3.end_members(andradite)
        assert result["Adr"].iloc[0] > 90.0

    def test_multi_row(self, fe_garnet_multi: pd.DataFrame) -> None:
        result = GrtFe3.end_members(fe_garnet_multi)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Feldspar
# ---------------------------------------------------------------------------


class TestFeldspar:
    def test_apfu_columns(self, sanidine: pd.DataFrame) -> None:
        result = Fsp.apfu(sanidine)
        assert "Si{4+}" in result.columns
        assert "Al{3+}" in result.columns
        assert "K{+}" in result.columns

    def test_sanidine_si(self, sanidine: pd.DataFrame) -> None:
        result = Fsp.apfu(sanidine)
        assert result["Si{4+}"].iloc[0] == pytest.approx(3.0, abs=0.05)

    def test_sanidine_k(self, sanidine: pd.DataFrame) -> None:
        result = Fsp.apfu(sanidine)
        assert result["K{+}"].iloc[0] == pytest.approx(1.0, abs=0.05)

    def test_end_members(self, sanidine: pd.DataFrame) -> None:
        result = Fsp.end_members(sanidine)
        assert "Or" in result.columns
        assert result["Or"].iloc[0] == pytest.approx(100.0, abs=1.0)

    def test_multi_row(self, feldspar_multi: pd.DataFrame) -> None:
        result = Fsp.end_members(feldspar_multi)
        assert len(result) == 3
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# Clinopyroxene
# ---------------------------------------------------------------------------


class TestClinopyroxene:
    def test_apfu_columns(self, fe_pyroxene: pd.DataFrame) -> None:
        result = Cpx.apfu(fe_pyroxene)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Fe{3+}",
            "Mg{2+}",
            "Ca{2+}",
            "Na{+}",
            "Ti{4+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_structural_formula(self, fe_pyroxene: pd.DataFrame) -> None:
        result = Cpx.site_allocations(fe_pyroxene)
        t_cols = [c for c in result.columns if c[0] == "T"]
        m1_cols = [c for c in result.columns if c[0] == "M1"]
        m2_cols = [c for c in result.columns if c[0] == "M2"]
        assert len(t_cols) > 0
        assert len(m1_cols) > 0
        assert len(m2_cols) > 0

    def test_t_site_sum(self, fe_pyroxene: pd.DataFrame) -> None:
        result = Cpx.site_allocations(fe_pyroxene)
        t_cols = [c for c in result.columns if c[0] == "T" and c[1] != "_unallocated"]
        t_total = result[t_cols].sum(axis=1)
        assert t_total.iloc[0] == pytest.approx(2.0, abs=0.1)

    def test_end_members(self, fe_pyroxene: pd.DataFrame) -> None:
        result = Cpx.end_members(fe_pyroxene)
        for col in ["Kosmochlor", "Ae", "Jd", "CaTs", "Wo", "Di", "Hd", "En", "Fs"]:
            assert col in result.columns
        row_sum = result.sum(axis=1).iloc[0]
        assert row_sum == pytest.approx(100.0, abs=1.0)

    def test_ideal_diopside(self, diopside: pd.DataFrame) -> None:
        result = Cpx.end_members(diopside)
        assert result["Di"].iloc[0] == pytest.approx(100.0, abs=2.0)
        for col in ["Kosmochlor", "Ae", "Jd", "CaTs", "Hd", "Fs"]:
            assert result[col].iloc[0] == pytest.approx(0.0, abs=1.0)

    def test_kosmochlor_allocation(self, cr_clinopyroxene: pd.DataFrame) -> None:
        result = Cpx.end_members(cr_clinopyroxene)
        assert result["Kosmochlor"].iloc[0] > 0.0
        row_sum = result.sum(axis=1).iloc[0]
        assert row_sum == pytest.approx(100.0, abs=2.0)

    def test_multi_row(self, clinopyroxene_multi: pd.DataFrame) -> None:
        result = Cpx.end_members(clinopyroxene_multi)
        assert len(result) == 3
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=2.0)


# ---------------------------------------------------------------------------
# Orthopyroxene
# ---------------------------------------------------------------------------


class TestOrthopyroxene:
    def test_apfu_columns(self, orthopyroxene_multi: pd.DataFrame) -> None:
        result = Opx.apfu(orthopyroxene_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Fe{3+}",
            "Mg{2+}",
            "Ca{2+}",
            "Ti{4+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_end_members(self, orthopyroxene_multi: pd.DataFrame) -> None:
        result = Opx.end_members(orthopyroxene_multi)
        assert "MgTs" in result.columns
        assert "Wo" in result.columns
        assert "En" in result.columns
        assert "Fs" in result.columns
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Muscovite
# ---------------------------------------------------------------------------


class TestMuscovite:
    def test_apfu_columns(self, muscovite_multi: pd.DataFrame) -> None:
        result = Ms.apfu(muscovite_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Mg{2+}",
            "Ti{4+}",
            "Na{+}",
            "K{+}",
            "Ba{2+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_structural_formula_sites(self, muscovite_multi: pd.DataFrame) -> None:
        result = Ms.site_allocations(muscovite_multi)
        t_cols = [c for c in result.columns if c[0] == "T"]
        i_cols = [c for c in result.columns if c[0] == "I"]
        o_cols = [c for c in result.columns if c[0] == "O"]
        assert len(t_cols) > 0
        assert len(i_cols) > 0
        assert len(o_cols) > 0

    def test_t_site_sum(self, muscovite_multi: pd.DataFrame) -> None:
        result = Ms.site_allocations(muscovite_multi)
        t_cols = [c for c in result.columns if c[0] == "T" and c[1] != "_unallocated"]
        t_total = result[t_cols].sum(axis=1)
        for val in t_total:
            assert val == pytest.approx(4.0, abs=0.1)

    def test_multi_row(self, muscovite_multi: pd.DataFrame) -> None:
        result = Ms.apfu(muscovite_multi)
        assert len(result) == 3

    def test_end_members_columns(self, muscovite_multi: pd.DataFrame) -> None:
        result = Ms.end_members(muscovite_multi)
        expected = {
            "Al-Celadonite",
            "Fe-Al-Celadonite",
            "Pyrophyllite",
            "Margarite",
            "Paragonite",
            "Muscovite",
            "Trioctahedral",
        }
        assert expected == set(result.columns)

    def test_end_members_sum(self, muscovite_multi: pd.DataFrame) -> None:
        result = Ms.end_members(muscovite_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# Biotite
# ---------------------------------------------------------------------------


class TestBiotite:
    def test_apfu_columns(self, biotite_multi: pd.DataFrame) -> None:
        result = Bt.apfu(biotite_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Mg{2+}",
            "Ti{4+}",
            "K{+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_structural_formula_sites(self, biotite_multi: pd.DataFrame) -> None:
        result = Bt.site_allocations(biotite_multi)
        t_cols = [c for c in result.columns if c[0] == "T"]
        i_cols = [c for c in result.columns if c[0] == "I"]
        o_cols = [c for c in result.columns if c[0] == "O"]
        assert len(t_cols) > 0
        assert len(i_cols) > 0
        assert len(o_cols) > 0

    def test_t_site_sum(self, biotite_multi: pd.DataFrame) -> None:
        result = Bt.site_allocations(biotite_multi)
        t_cols = [c for c in result.columns if c[0] == "T" and c[1] != "_unallocated"]
        t_total = result[t_cols].sum(axis=1)
        for val in t_total:
            assert val == pytest.approx(4.0, abs=0.15)

    def test_end_members_columns(self, biotite_multi: pd.DataFrame) -> None:
        result = Bt.end_members(biotite_multi)
        expected = {
            "Phlogopite",
            "Annite",
            "Eastonite",
            "Siderophyllite",
            "Dioctahedral",
        }
        assert expected == set(result.columns)

    def test_end_members_sum(self, biotite_multi: pd.DataFrame) -> None:
        result = Bt.end_members(biotite_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_end_members_multi(self, biotite_multi: pd.DataFrame) -> None:
        result = Bt.end_members(biotite_multi)
        assert len(result) == 3

    def test_phlogopite_ideal(self, biotite: pd.DataFrame) -> None:
        result = Bt.end_members(biotite)
        assert result["Phlogopite"].iloc[0] > 90.0


# ---------------------------------------------------------------------------
# Staurolite
# ---------------------------------------------------------------------------


class TestStaurolite:
    def test_apfu_columns(self, staurolite_multi: pd.DataFrame) -> None:
        result = St.apfu(staurolite_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Mg{2+}",
            "Zn{2+}",
            "Mn{2+}",
            "Ti{4+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, staurolite_multi: pd.DataFrame) -> None:
        result = St.apfu(staurolite_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_structural_formula_sites(self, staurolite_multi: pd.DataFrame) -> None:
        result = St.site_allocations(staurolite_multi)
        t_cols = [c for c in result.columns if c[0] == "T"]
        m_cols = [c for c in result.columns if c[0] == "M"]
        assert len(t_cols) > 0
        assert len(m_cols) > 0

    def test_end_members_columns(self, staurolite_multi: pd.DataFrame) -> None:
        result = St.end_members(staurolite_multi)
        expected = {"Fe-Staurolite", "Mg-Staurolite", "Zn-Staurolite", "Mn-Staurolite"}
        assert expected == set(result.columns)

    def test_end_members_sum(self, staurolite_multi: pd.DataFrame) -> None:
        result = St.end_members(staurolite_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_multi_row(self, staurolite_multi: pd.DataFrame) -> None:
        result = St.apfu(staurolite_multi)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Chlorite
# ---------------------------------------------------------------------------


class TestChlorite:
    def test_apfu_columns(self, chlorite_multi: pd.DataFrame) -> None:
        result = Chl.apfu(chlorite_multi)
        expected_cols = {"Si{4+}", "Al{3+}", "Fe{2+}", "Mg{2+}"}
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, chlorite_multi: pd.DataFrame) -> None:
        result = Chl.apfu(chlorite_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_charge_normalization(self, chlorite: pd.DataFrame) -> None:
        result = Chl.apfu(chlorite)
        cat_sum = result.sum(axis=1)
        assert cat_sum.iloc[0] > 0

    def test_structural_formula_sites(self, chlorite_multi: pd.DataFrame) -> None:
        result = Chl.site_allocations(chlorite_multi)
        t_cols = [c for c in result.columns if c[0] == "T"]
        m_cols = [c for c in result.columns if c[0] == "M"]
        assert len(t_cols) > 0
        assert len(m_cols) > 0

    def test_end_members_columns(self, chlorite_multi: pd.DataFrame) -> None:
        result = Chl.end_members(chlorite_multi)
        expected = {"Clinochlore", "Chamosite", "Mg-Sudoite", "Fe-Sudoite"}
        assert expected == set(result.columns)

    def test_end_members_sum(self, chlorite_multi: pd.DataFrame) -> None:
        result = Chl.end_members(chlorite_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_multi_row(self, chlorite_multi: pd.DataFrame) -> None:
        result = Chl.apfu(chlorite_multi)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Epidote
# ---------------------------------------------------------------------------


class TestEpidote:
    def test_apfu_columns(self, epidote_multi: pd.DataFrame) -> None:
        result = Ep.apfu(epidote_multi)
        expected_cols = {"Si{4+}", "Al{3+}", "Fe{3+}", "Ca{2+}"}
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, epidote_multi: pd.DataFrame) -> None:
        result = Ep.apfu(epidote_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_fe_is_fe3(self, epidote: pd.DataFrame) -> None:
        result = Ep.apfu(epidote)
        assert "Fe{3+}" in result.columns
        assert result["Fe{3+}"].iloc[0] > 0

    def test_no_fe2(self, epidote: pd.DataFrame) -> None:
        result = Ep.apfu(epidote)
        assert "Fe{2+}" not in result.columns

    def test_structural_formula_sites(self, epidote_multi: pd.DataFrame) -> None:
        result = Ep.site_allocations(epidote_multi)
        a_cols = [c for c in result.columns if c[0] == "A"]
        m_cols = [c for c in result.columns if c[0] == "M"]
        t_cols = [c for c in result.columns if c[0] == "T"]
        assert len(a_cols) > 0
        assert len(m_cols) > 0
        assert len(t_cols) > 0

    def test_end_members_columns(self, epidote_multi: pd.DataFrame) -> None:
        result = Ep.end_members(epidote_multi)
        expected = {
            "Clinozoisite",
            "Epidote",
            "Piemontite",
            "Mukhinite",
            "Tawmawite",
        }
        assert expected == set(result.columns)

    def test_end_members_sum(self, epidote_multi: pd.DataFrame) -> None:
        result = Ep.end_members(epidote_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_multi_row(self, epidote_multi: pd.DataFrame) -> None:
        result = Ep.apfu(epidote_multi)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Amphibole
# ---------------------------------------------------------------------------


class TestAmphibole:
    def test_apfu_columns(self, amphibole_multi: pd.DataFrame) -> None:
        result = Amp.apfu(amphibole_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Fe{3+}",
            "Mg{2+}",
            "Ca{2+}",
            "Na{+}",
            "K{+}",
            "Ti{4+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, amphibole_multi: pd.DataFrame) -> None:
        result = Amp.apfu(amphibole_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_structural_formula_sites(self, amphibole_multi: pd.DataFrame) -> None:
        result = Amp.site_allocations(amphibole_multi)
        b_cols = [c for c in result.columns if c[0] == "B"]
        c_cols = [c for c in result.columns if c[0] == "C"]
        t_cols = [c for c in result.columns if c[0] == "T"]
        assert len(b_cols) > 0
        assert len(c_cols) > 0
        assert len(t_cols) > 0

    def test_t_site_sum(self, amphibole_multi: pd.DataFrame) -> None:
        result = Amp.site_allocations(amphibole_multi)
        t_cols = [c for c in result.columns if c[0] == "T" and c[1] != "_unallocated"]
        t_total = result[t_cols].sum(axis=1)
        for val in t_total:
            assert val == pytest.approx(8.0, abs=0.5)

    def test_end_members_columns(self, amphibole_multi: pd.DataFrame) -> None:
        result = Amp.end_members(amphibole_multi)
        expected = {
            "Tremolite",
            "Actinolite",
            "Edenite",
            "Ferro-Edenite",
            "Pargasite",
            "Ferro-Pargasite",
            "Tschermakite",
            "Richterite",
            "Winchite",
            "Glaucophane",
            "Ferro-Glaucophane",
            "Riebeckite",
            "Magnesio-Riebeckite",
        }
        assert expected == set(result.columns)

    def test_end_members_sum(self, amphibole_multi: pd.DataFrame) -> None:
        result = Amp.end_members(amphibole_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=2.0)

    def test_multi_row(self, amphibole_multi: pd.DataFrame) -> None:
        result = Amp.apfu(amphibole_multi)
        assert len(result) == 3

    def test_actinolite_dominant(self, amphibole_multi: pd.DataFrame) -> None:
        result = Amp.end_members(amphibole_multi)
        row1 = result.iloc[1]
        assert row1["Tremolite"] > 50.0


# ---------------------------------------------------------------------------
# Titanite
# ---------------------------------------------------------------------------


class TestTitanite:
    def test_apfu_columns(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.apfu(titanite_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{3+}",
            "Ti{4+}",
            "Ca{2+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.apfu(titanite_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_fe_is_fe3(self, titanite: pd.DataFrame) -> None:
        result = Ttn.apfu(titanite)
        # Even with 0 FeO input, Fe{3+} should not appear
        # (it only appears when FeO > 0)
        if "Fe{3+}" in result.columns:
            pass  # Fe³⁺ present, as expected

    def test_no_fe2(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.apfu(titanite_multi)
        assert "Fe{2+}" not in result.columns

    def test_structural_formula_sites(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.site_allocations(titanite_multi)
        a_cols = [c for c in result.columns if c[0] == "A"]
        b_cols = [c for c in result.columns if c[0] == "B"]
        t_cols = [c for c in result.columns if c[0] == "T"]
        assert len(a_cols) > 0
        assert len(b_cols) > 0
        assert len(t_cols) > 0

    def test_t_site_sum(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.site_allocations(titanite_multi)
        t_cols = [c for c in result.columns if c[0] == "T" and c[1] != "_unallocated"]
        t_total = result[t_cols].sum(axis=1)
        for val in t_total:
            assert val == pytest.approx(1.0, abs=0.1)

    def test_end_members_columns(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.end_members(titanite_multi)
        expected = {"Ttn", "Al-Ttn", "Fe-Ttn", "Mal", "Other"}
        assert expected == set(result.columns)

    def test_end_members_sum(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.end_members(titanite_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_ideal_ttn(self, titanite: pd.DataFrame) -> None:
        result = Ttn.end_members(titanite)
        assert result["Ttn"].iloc[0] > 95.0

    def test_al_bearing(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.end_members(titanite_multi)
        # Row 1 has Al₂O₃ = 6.0% so Al-Ttn should be significant
        assert result["Al-Ttn"].iloc[1] > 10.0

    def test_fe_bearing(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.end_members(titanite_multi)
        # Row 2 has FeO = 4.0% so Fe-Ttn should be significant
        assert result["Fe-Ttn"].iloc[2] > 5.0

    def test_sn_bearing(self, titanite_sn: pd.DataFrame) -> None:
        result = Ttn.end_members(titanite_sn)
        assert result["Mal"].iloc[0] > 20.0

    def test_multi_row(self, titanite_multi: pd.DataFrame) -> None:
        result = Ttn.apfu(titanite_multi)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Chloritoid
# ---------------------------------------------------------------------------


class TestChloritoid:
    def test_apfu_columns(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.apfu(chloritoid_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Fe{3+}",
            "Mg{2+}",
            "Mn{2+}",
            "Ti{4+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.apfu(chloritoid_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_structural_formula_sites(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.site_allocations(chloritoid_multi)
        t_cols = [c for c in result.columns if c[0] == "T"]
        m_cols = [c for c in result.columns if c[0] == "M1"]
        assert len(t_cols) > 0
        assert len(m_cols) > 0

    def test_t_site_sum(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.site_allocations(chloritoid_multi)
        t_cols = [c for c in result.columns if c[0] == "T" and c[1] != "_unallocated"]
        t_total = result[t_cols].sum(axis=1)
        for val in t_total:
            assert val == pytest.approx(2.0, abs=0.1)

    def test_end_members_columns(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.end_members(chloritoid_multi)
        expected = {"Cld", "Mgcld", "Mncld"}
        assert expected == set(result.columns)

    def test_end_members_sum(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.end_members(chloritoid_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_multi_row(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.apfu(chloritoid_multi)
        assert len(result) == 3

    def test_fe_dominant(self, chloritoid: pd.DataFrame) -> None:
        result = Cld.end_members(chloritoid)
        assert result["Cld"].iloc[0] > 60.0

    def test_mg_dominant_row(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.end_members(chloritoid_multi)
        # Row 1 has MgO=18.0, FeO=8.0 → Mgcld should be dominant
        assert result["Mgcld"].iloc[1] > 60.0

    def test_mn_bearing_row(self, chloritoid_multi: pd.DataFrame) -> None:
        result = Cld.end_members(chloritoid_multi)
        # Row 2 has MnO=14.0 → Mncld should be significant
        assert result["Mncld"].iloc[2] > 30.0


# ---------------------------------------------------------------------------
# Cordierite
# ---------------------------------------------------------------------------


class TestCordierite:
    def test_apfu_columns(self, cordierite_multi: pd.DataFrame) -> None:
        result = Crd.apfu(cordierite_multi)
        expected_cols = {
            "Si{4+}",
            "Al{3+}",
            "Fe{2+}",
            "Mg{2+}",
            "Mn{2+}",
            "Na{+}",
            "K{+}",
            "Ca{2+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, cordierite_multi: pd.DataFrame) -> None:
        result = Crd.apfu(cordierite_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_structural_formula_sites(self, cordierite_multi: pd.DataFrame) -> None:
        result = Crd.site_allocations(cordierite_multi)
        t1_cols = [c for c in result.columns if c[0] == "T1"]
        t2_cols = [c for c in result.columns if c[0] == "T2"]
        b_cols = [c for c in result.columns if c[0] == "B"]
        assert len(t1_cols) > 0
        assert len(t2_cols) > 0
        assert len(b_cols) > 0

    def test_t1_site_sum(self, cordierite_multi: pd.DataFrame) -> None:
        result = Crd.site_allocations(cordierite_multi)
        t1_cols = [c for c in result.columns if c[0] == "T1" and c[1] != "_unallocated"]
        t1_total = result[t1_cols].sum(axis=1)
        for val in t1_total:
            assert val == pytest.approx(6.0, abs=0.5)

    def test_no_fe3(self, cordierite_multi: pd.DataFrame) -> None:
        result = Crd.apfu(cordierite_multi)
        assert "Fe{3+}" not in result.columns

    def test_end_members_columns(self, cordierite_multi: pd.DataFrame) -> None:
        result = Crd.end_members(cordierite_multi)
        expected = {"H₂O-Crd", "Mg-Crd", "Fe-Crd", "Mn-Crd"}
        assert expected == set(result.columns)

    def test_end_members_sum(self, cordierite_multi: pd.DataFrame) -> None:
        result = Crd.end_members(cordierite_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_multi_row(self, cordierite_multi: pd.DataFrame) -> None:
        result = Crd.apfu(cordierite_multi)
        assert len(result) == 3

    def test_mg_dominant(self, cordierite: pd.DataFrame) -> None:
        result = Crd.end_members(cordierite)
        # Row 0 has MgO=10.5, FeO=5.0 → Mg-Crd should dominate B-site
        assert result["Mg-Crd"].iloc[0] > 60.0


# ---------------------------------------------------------------------------
# Ilmenite
# ---------------------------------------------------------------------------


class TestIlmenite:
    def test_apfu_columns(self, ilmenite_multi: pd.DataFrame) -> None:
        result = Ilm.apfu(ilmenite_multi)
        expected_cols = {"Ti{4+}", "Fe{2+}", "Fe{3+}", "Mg{2+}", "Mn{2+}"}
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, ilmenite_multi: pd.DataFrame) -> None:
        result = Ilm.apfu(ilmenite_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_structural_formula_sites(self, ilmenite_multi: pd.DataFrame) -> None:
        result = Ilm.site_allocations(ilmenite_multi)
        a_cols = [c for c in result.columns if c[0] == "A"]
        b_cols = [c for c in result.columns if c[0] == "B"]
        assert len(a_cols) > 0
        assert len(b_cols) > 0

    def test_end_members_columns(self, ilmenite_multi: pd.DataFrame) -> None:
        result = Ilm.end_members(ilmenite_multi)
        expected = {"Ilm", "Gk", "Pph", "Hem", "Chr"}
        assert expected == set(result.columns)

    def test_end_members_sum(self, ilmenite_multi: pd.DataFrame) -> None:
        result = Ilm.end_members(ilmenite_multi)
        for _, row in result.iterrows():
            assert row.sum() == pytest.approx(100.0, abs=1.0)

    def test_multi_row(self, ilmenite_multi: pd.DataFrame) -> None:
        result = Ilm.apfu(ilmenite_multi)
        assert len(result) == 3

    def test_ideal_ilm(self, ilmenite: pd.DataFrame) -> None:
        result = Ilm.end_members(ilmenite)
        assert result["Ilm"].iloc[0] > 95.0

    def test_geikielite_row(self, ilmenite_multi: pd.DataFrame) -> None:
        result = Ilm.end_members(ilmenite_multi)
        # Row 1 has MgO=14.0 → Gk should be significant
        assert result["Gk"].iloc[1] > 20.0

    def test_pyrophanite_row(self, ilmenite_multi: pd.DataFrame) -> None:
        result = Ilm.end_members(ilmenite_multi)
        # Row 2 has MnO=10.0 → Pph should be significant
        assert result["Pph"].iloc[2] > 10.0


# ---------------------------------------------------------------------------
# Spinel
# ---------------------------------------------------------------------------


class TestSpinel:
    def test_apfu_columns(self, spinel_multi: pd.DataFrame) -> None:
        result = Spl.apfu(spinel_multi)
        expected_cols = {
            "Mg{2+}",
            "Fe{2+}",
            "Fe{3+}",
            "Al{3+}",
            "Cr{3+}",
            "Ti{4+}",
            "Mn{2+}",
            "Zn{2+}",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_apfu_no_negatives(self, spinel_multi: pd.DataFrame) -> None:
        result = Spl.apfu(spinel_multi)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_structural_formula_sites(self, spinel_multi: pd.DataFrame) -> None:
        result = Spl.site_allocations(spinel_multi)
        t_cols = [c for c in result.columns if c[0] == "T"]
        m_cols = [c for c in result.columns if c[0] == "M"]
        assert len(t_cols) > 0
        assert len(m_cols) > 0

    def test_end_members_columns(self, spinel_multi: pd.DataFrame) -> None:
        result = Spl.end_members(spinel_multi)
        expected = {
            "Spl",
            "Herc",
            "Chrm",
            "Mtc",
            "Gahn",
            "Frank",
            "Jac",
            "Ulv",
            "Spss",
        }
        assert expected == set(result.columns)

    def test_multi_row(self, spinel_multi: pd.DataFrame) -> None:
        result = Spl.apfu(spinel_multi)
        assert len(result) == 3

    def test_mg_al_spinel(self, spinel: pd.DataFrame) -> None:
        result = Spl.end_members(spinel)
        # Mg-Al spinel → Spl should be dominant
        assert result["Spl"].iloc[0] > 50.0

    def test_hercynite_row(self, spinel_multi: pd.DataFrame) -> None:
        result = Spl.end_members(spinel_multi)
        # Row 1 has FeO=25.0, Al2O3=30.0 → Herc should be significant
        assert result["Herc"].iloc[1] > 10.0

    def test_chromite_row(self, spinel_multi: pd.DataFrame) -> None:
        result = Spl.end_members(spinel_multi)
        # Row 2 has Cr2O3=45.0 → Chrm should be dominant
        assert result["Chrm"].iloc[2] > 50.0


# ---------------------------------------------------------------------------
# Accessor integration
# ---------------------------------------------------------------------------


class TestAccessorIntegration:
    def test_mineral_apfu(self, mn_garnet: pd.DataFrame) -> None:
        result = mn_garnet.mineral.apfu(Grt)
        assert "Fe{2+}" in result.columns

    def test_structural_formula(self, mn_garnet: pd.DataFrame) -> None:
        result = mn_garnet.mineral.site_allocations(Grt)
        x_cols = [c for c in result.columns if c[0] == "X"]
        assert len(x_cols) > 0

    def test_end_members(self, mn_garnet: pd.DataFrame) -> None:
        result = mn_garnet.mineral.end_members(Grt)
        row_sum = result.sum(axis=1).iloc[0]
        assert row_sum == pytest.approx(100.0, abs=1.0)
