"""Tests for check_stoichiometry and _score_trapezoidal."""

from __future__ import annotations

import pandas as pd

from petropandas import (
    Amp,
    Bt,
    Chl,
    Cld,
    Cpx,
    Crd,
    Ep,
    Fsp,
    Grt,
    Ilm,
    Ms,
    Opx,
    Spl,
    St,
    Ttn,
)
from petropandas._minerals import _score_trapezoidal


# ---------------------------------------------------------------------------
# _score_trapezoidal unit tests
# ---------------------------------------------------------------------------


class TestScoreTrapezoidal:
    def test_inside_ideal_returns_1(self):
        assert _score_trapezoidal(100.0, 99.0, 101.0) == 1.0
        assert _score_trapezoidal(99.0, 99.0, 101.0) == 1.0
        assert _score_trapezoidal(101.0, 99.0, 101.0) == 1.0

    def test_at_decay_edge_returns_0(self):
        assert _score_trapezoidal(97.5, 99.0, 101.0) == 0.0
        assert _score_trapezoidal(102.5, 99.0, 101.0) == 0.0

    def test_beyond_decay_edge_returns_0(self):
        assert _score_trapezoidal(95.0, 99.0, 101.0) == 0.0
        assert _score_trapezoidal(105.0, 99.0, 101.0) == 0.0

    def test_midpoint_of_decay_is_0_5(self):
        score = _score_trapezoidal(98.25, 99.0, 101.0)
        assert abs(score - 0.5) < 1e-10
        score = _score_trapezoidal(101.75, 99.0, 101.0)
        assert abs(score - 0.5) < 1e-10

    def test_custom_margin(self):
        assert _score_trapezoidal(98.0, 99.0, 101.0, margin=1.0) == 0.0
        assert _score_trapezoidal(98.0, 99.0, 101.0, margin=2.0) == 0.5


# ---------------------------------------------------------------------------
# check_stoichiometry — shape and columns
# ---------------------------------------------------------------------------


class TestCheckStoichiometryShape:
    def test_garnet_returns_correct_shape(self, garnet_multi):
        result = garnet_multi.mineral.check_stoichiometry(Grt)
        assert result.shape == (3, 6)

    def test_clinopyroxene_has_fe3_column(self, fe_pyroxene):
        result = fe_pyroxene.mineral.check_stoichiometry(Cpx)
        assert "fe3+_validity" in result.columns

    def test_feldspar_has_tetrahedral_fill(self, feldspar_multi):
        result = feldspar_multi.mineral.check_stoichiometry(Fsp)
        assert "tetrahedral_fill" in result.columns

    def test_no_fe_split_excludes_fe3_column(self, sanidine):
        result = sanidine.mineral.check_stoichiometry(Fsp)
        assert "fe3+_validity" not in result.columns

    def test_no_ideal_cations_excludes_cation_deviation(self, staurolite):
        result = staurolite.mineral.check_stoichiometry(St)
        assert "cation_deviation" not in result.columns

    def test_no_t_site_excludes_tetrahedral_fill(self, garnet_multi):
        result = garnet_multi.mineral.check_stoichiometry(Grt)
        assert "tetrahedral_fill" not in result.columns


# ---------------------------------------------------------------------------
# check_stoichiometry — scoring behavior
# ---------------------------------------------------------------------------


class TestCheckStoichiometryScoring:
    def test_garnet_analytical_total_in_range(self, garnet_multi):
        result = garnet_multi.mineral.check_stoichiometry(Grt)
        totals = garnet_multi[["SiO2", "Al2O3", "FeO", "MnO", "MgO", "CaO"]].sum(axis=1)
        for i, total in enumerate(totals):
            if 99.0 <= total <= 101.0:
                assert result.loc[i, "analytical_total"] == 1.0

    def test_cation_deviation_near_1_for_ideal(self, diopside):
        result = diopside.mineral.check_stoichiometry(Cpx)
        assert result.loc[0, "cation_deviation"] > 0.9

    def test_charge_balance_high_for_valid_analysis(self, diopside):
        result = diopside.mineral.check_stoichiometry(Cpx)
        assert result.loc[0, "charge_balance"] > 0.8

    def test_fe3_validity_is_1_when_no_negative(self, fe_pyroxene):
        result = fe_pyroxene.mineral.check_stoichiometry(Cpx)
        assert result.loc[0, "fe3+_validity"] == 1.0

    def test_all_scores_between_0_and_1(self, fe_garnet_multi):
        result = fe_garnet_multi.mineral.check_stoichiometry(Grt)
        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} has values < 0"
            assert (result[col] <= 1).all(), f"{col} has values > 1"

    def test_multi_row_preserves_index(self, garnet_multi):
        result = garnet_multi.mineral.check_stoichiometry(Grt)
        pd.testing.assert_index_equal(result.index, garnet_multi.index)


# ---------------------------------------------------------------------------
# check_stoichiometry — per-mineral integration
# ---------------------------------------------------------------------------


class TestCheckStoichiometryMinerals:
    def test_amphibole(self, amphibole):
        result = amphibole.mineral.check_stoichiometry(Amp)
        assert result.shape[0] == 1
        assert "analytical_total" in result.columns
        assert "cation_deviation" in result.columns

    def test_biotite(self, biotite):
        result = biotite.mineral.check_stoichiometry(Bt)
        assert result.shape[0] == 1
        assert "analytical_total" in result.columns

    def test_chlorite(self, chlorite):
        result = chlorite.mineral.check_stoichiometry(Chl)
        assert result.shape[0] == 1
        assert "analytical_total" in result.columns
        assert "cation_deviation" not in result.columns

    def test_chloritoid(self, chloritoid):
        result = chloritoid.mineral.check_stoichiometry(Cld)
        assert result.shape[0] == 1
        assert "fe3+_validity" in result.columns

    def test_cordierite(self, cordierite):
        result = cordierite.mineral.check_stoichiometry(Crd)
        assert result.shape[0] == 1
        assert "tetrahedral_fill" in result.columns

    def test_epidote(self, epidote):
        result = epidote.mineral.check_stoichiometry(Ep)
        assert result.shape[0] == 1
        assert "fe3+_validity" in result.columns  # FeO→Fe₂O₃ gives Fe{3+}

    def test_ilmenite(self, ilmenite):
        result = ilmenite.mineral.check_stoichiometry(Ilm)
        assert result.shape[0] == 1
        assert "fe3+_validity" in result.columns
        assert "tetrahedral_fill" not in result.columns  # no T-site

    def test_muscovite(self, muscovite_multi):
        result = muscovite_multi.mineral.check_stoichiometry(Ms)
        assert result.shape[0] == 3
        assert "tetrahedral_fill" in result.columns

    def test_orthopyroxene(self, orthopyroxene_multi):
        result = orthopyroxene_multi.mineral.check_stoichiometry(Opx)
        assert result.shape[0] == 3
        assert "tetrahedral_fill" in result.columns

    def test_spinel(self, spinel):
        result = spinel.mineral.check_stoichiometry(Spl)
        assert result.shape[0] == 1
        assert "fe3+_validity" in result.columns

    def test_staurolite(self, staurolite):
        result = staurolite.mineral.check_stoichiometry(St)
        assert result.shape[0] == 1
        assert "tetrahedral_fill" in result.columns

    def test_titanite(self, titanite):
        result = titanite.mineral.check_stoichiometry(Ttn)
        assert result.shape[0] == 1
        assert "analytical_total" in result.columns

    def test_multi_row_amphibole(self, amphibole_multi):
        result = amphibole_multi.mineral.check_stoichiometry(Amp)
        assert result.shape == (3, 7)

    def test_multi_row_clinopyroxene(self, clinopyroxene_multi):
        result = clinopyroxene_multi.mineral.check_stoichiometry(Cpx)
        assert result.shape == (3, 7)

    def test_multi_row_feldspar(self, feldspar_multi):
        result = feldspar_multi.mineral.check_stoichiometry(Fsp)
        assert result.shape == (3, 6)

    def test_multi_row_biotite(self, biotite_multi):
        result = biotite_multi.mineral.check_stoichiometry(Bt)
        assert result.shape == (3, 6)

    def test_multi_row_chlorite(self, chlorite_multi):
        result = chlorite_multi.mineral.check_stoichiometry(Chl)
        assert result.shape == (3, 5)

    def test_multi_row_epidote(self, epidote_multi):
        result = epidote_multi.mineral.check_stoichiometry(Ep)
        assert result.shape == (3, 7)

    def test_multi_row_titanite(self, titanite_multi):
        result = titanite_multi.mineral.check_stoichiometry(Ttn)
        assert result.shape == (3, 7)

    def test_multi_row_chloritoid(self, chloritoid_multi):
        result = chloritoid_multi.mineral.check_stoichiometry(Cld)
        assert result.shape == (3, 7)

    def test_multi_row_cordierite(self, cordierite_multi):
        result = cordierite_multi.mineral.check_stoichiometry(Crd)
        assert result.shape == (3, 6)

    def test_multi_row_ilmenite(self, ilmenite_multi):
        result = ilmenite_multi.mineral.check_stoichiometry(Ilm)
        assert result.shape == (3, 6)

    def test_multi_row_spinel(self, spinel_multi):
        result = spinel_multi.mineral.check_stoichiometry(Spl)
        assert result.shape == (3, 7)


# ---------------------------------------------------------------------------
# check_stoichiometry — analytical_total_range on minerals
# ---------------------------------------------------------------------------


class TestAnalyticalTotalRange:
    def test_garnet_range(self):
        assert Grt.analytical_total_range == (99.0, 101.0)

    def test_chlorite_range(self):
        assert Chl.analytical_total_range == (85.0, 90.0)

    def test_amphibole_range(self):
        assert Amp.analytical_total_range == (96.0, 99.0)

    def test_spinel_range(self):
        assert Spl.analytical_total_range == (93.0, 100.5)

    def test_cordierite_range(self):
        assert Crd.analytical_total_range == (97.0, 99.0)

    def test_muscovite_range(self):
        assert Ms.analytical_total_range == (94.0, 97.0)

    def test_biotite_range(self):
        assert Bt.analytical_total_range == (94.0, 97.0)
