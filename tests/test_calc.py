"""Tests for _calc.py — pure calculation functions."""

from __future__ import annotations

import pandas as pd
import pytest

from petropandas._calc import (
    cation_moles,
    convert,
    fe2o3_to_feo,
    feo_to_fe2o3,
    from_apfu,
    molecular_weights,
    normalize,
    oxidize_moles,
    oxygen_moles,
    reduce_moles,
    split_valence,
    to_apfu,
    to_moles,
    to_oxides,
)
from petropandas._core import MW


class TestToMoles:
    def test_diopside(self, diopside: pd.DataFrame) -> None:
        result = to_moles(diopside)
        assert list(result.columns) == ["SiO2", "MgO", "CaO"]
        assert result["SiO2"].iloc[0] == pytest.approx(0.9235, abs=0.001)

    def test_columns_match_input(self, fe_pyroxene: pd.DataFrame) -> None:
        result = to_moles(fe_pyroxene)
        assert set(result.columns) == set(fe_pyroxene.columns)


class TestToOxides:
    def test_roundtrip(self, diopside: pd.DataFrame) -> None:
        moles = to_moles(diopside)
        back = to_oxides(moles)
        pd.testing.assert_frame_equal(back, diopside[["SiO2", "MgO", "CaO"]])

    def test_roundtrip_fe(self, fe_pyroxene: pd.DataFrame) -> None:
        moles = to_moles(fe_pyroxene)
        back = to_oxides(moles)
        pd.testing.assert_frame_equal(back, fe_pyroxene)


class TestCationMoles:
    def test_diopside_si(self, diopside: pd.DataFrame) -> None:
        result = cation_moles(diopside)
        assert result["SiO2"].iloc[0] == pytest.approx(0.9235, abs=0.001)

    def test_diopside_ca(self, diopside: pd.DataFrame) -> None:
        result = cation_moles(diopside)
        assert result["CaO"].iloc[0] == pytest.approx(0.4618, abs=0.001)


class TestOxygenMoles:
    def test_diopside_si(self, diopside: pd.DataFrame) -> None:
        result = oxygen_moles(diopside)
        assert result["SiO2"].iloc[0] == pytest.approx(1.847, abs=0.002)


class TestToApfu:
    def test_diopside_oxygen_basis(self, diopside: pd.DataFrame) -> None:
        result = to_apfu(diopside, n_oxygens=6)
        assert "Si{4+}" in result.columns
        assert "Mg{2+}" in result.columns
        assert "Ca{2+}" in result.columns
        assert result["Si{4+}"].iloc[0] == pytest.approx(2.00, abs=0.01)
        assert result["Mg{2+}"].iloc[0] == pytest.approx(1.00, abs=0.01)
        assert result["Ca{2+}"].iloc[0] == pytest.approx(1.00, abs=0.01)

    def test_diopside_cation_basis(self, diopside: pd.DataFrame) -> None:
        result = to_apfu(diopside, n_cations=4)
        assert result["Si{4+}"].iloc[0] == pytest.approx(2.00, abs=0.01)
        assert result["Mg{2+}"].iloc[0] == pytest.approx(1.00, abs=0.01)
        assert result["Ca{2+}"].iloc[0] == pytest.approx(1.00, abs=0.01)

    def test_exactly_one_param(self, diopside: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            to_apfu(diopside)
        with pytest.raises(ValueError, match="exactly one"):
            to_apfu(diopside, n_oxygens=6, n_cations=4)

    def test_sanidine(self, sanidine: pd.DataFrame) -> None:
        result = to_apfu(sanidine, n_oxygens=8)
        assert result["Si{4+}"].iloc[0] == pytest.approx(3.00, abs=0.01)
        assert result["Al{3+}"].iloc[0] == pytest.approx(1.00, abs=0.01)
        assert result["K{+}"].iloc[0] == pytest.approx(1.00, abs=0.01)

    def test_from_moles(self, diopside: pd.DataFrame) -> None:
        from_wt = to_apfu(diopside, n_oxygens=6)
        moles = to_moles(diopside)
        from_moles = to_apfu(moles, n_oxygens=6, units="moles")
        pd.testing.assert_frame_equal(from_wt, from_moles)

    def test_ion_column_names(self, fe_pyroxene: pd.DataFrame) -> None:
        result = to_apfu(fe_pyroxene, n_oxygens=6)
        for col in result.columns:
            # All columns should be ion notation
            assert "{" in col and "}" in col, f"Expected ion column name, got {col!r}"


class TestNormalize:
    def test_sums_to_100(self, fe_pyroxene: pd.DataFrame) -> None:
        result = normalize(fe_pyroxene)
        assert result.sum(axis=1).iloc[0] == pytest.approx(100.0)


class TestSplitValence:
    def test_droop_fe(self, fe_pyroxene: pd.DataFrame) -> None:
        apfu = to_apfu(fe_pyroxene, n_oxygens=6)
        result = split_valence(apfu, "Fe", method="droop", n_oxygens=6, ideal_cations=4)
        assert "Fe{3+}" in result.columns
        assert "Fe{2+}" in result.columns
        assert (result["Fe{2+}"] >= 0).all()
        assert (result["Fe{3+}"] >= 0).all()

    def test_schumacher_fe(self, fe_pyroxene: pd.DataFrame) -> None:
        apfu = to_apfu(fe_pyroxene, n_oxygens=6)
        result = split_valence(
            apfu, "Fe", method="schumacher", n_oxygens=6, ideal_cations=4
        )
        assert "Fe{3+}" in result.columns
        assert "Fe{2+}" in result.columns
        assert (result["Fe{2+}"] >= 0).all()
        assert (result["Fe{3+}"] >= 0).all()

    def test_droop_mn(self, mn_garnet: pd.DataFrame) -> None:
        apfu = to_apfu(mn_garnet, n_oxygens=12)
        result = split_valence(
            apfu, "Mn", method="droop", n_oxygens=12, ideal_cations=8
        )
        assert "Mn{3+}" in result.columns
        assert "Mn{2+}" in result.columns
        assert (result["Mn{2+}"] >= 0).all()
        assert (result["Mn{3+}"] >= 0).all()

    def test_schumacher_mn(self, mn_garnet: pd.DataFrame) -> None:
        apfu = to_apfu(mn_garnet, n_oxygens=12)
        result = split_valence(
            apfu, "Mn", method="schumacher", n_oxygens=12, ideal_cations=8
        )
        assert "Mn{3+}" in result.columns
        assert "Mn{2+}" in result.columns
        assert (result["Mn{2+}"] >= 0).all()
        assert (result["Mn{3+}"] >= 0).all()

    def test_droop_ti(self, ti_rutile: pd.DataFrame) -> None:
        apfu = to_apfu(ti_rutile, n_oxygens=6)
        result = split_valence(apfu, "Ti", method="droop", n_oxygens=6, ideal_cations=4)
        assert "Ti{3+}" in result.columns
        assert "Ti{4+}" in result.columns
        assert (result["Ti{4+}"] >= 0).all()
        assert (result["Ti{3+}"] >= 0).all()

    def test_schumacher_ti(self, ti_rutile: pd.DataFrame) -> None:
        apfu = to_apfu(ti_rutile, n_oxygens=6)
        result = split_valence(
            apfu, "Ti", method="schumacher", n_oxygens=6, ideal_cations=4
        )
        assert "Ti{3+}" in result.columns
        assert "Ti{4+}" in result.columns
        assert (result["Ti{4+}"] >= 0).all()
        assert (result["Ti{3+}"] >= 0).all()

    def test_unknown_element_raises(self, fe_pyroxene: pd.DataFrame) -> None:
        apfu = to_apfu(fe_pyroxene, n_oxygens=6)
        with pytest.raises(ValueError, match="Unknown element"):
            split_valence(apfu, "X", method="droop", n_oxygens=6, ideal_cations=4)

    def test_invalid_method_raises(self, fe_pyroxene: pd.DataFrame) -> None:
        apfu = to_apfu(fe_pyroxene, n_oxygens=6)
        with pytest.raises(ValueError, match="Unknown method"):
            split_valence(apfu, "Fe", method="bad", n_oxygens=6, ideal_cations=4)


class TestOxidizeMoles:
    def test_basic_split(self, fe_pyroxene: pd.DataFrame) -> None:
        moles = to_moles(fe_pyroxene)
        total_feo = moles["FeO"].copy()
        result = oxidize_moles(moles, o_excess=0.05)
        assert "FeO" in result.columns
        assert "Fe2O3" in result.columns
        assert (result["FeO"] >= 0).all()
        assert (result["Fe2O3"] >= 0).all()
        fe2_remaining = result["FeO"]
        fe3_as_feo = 2.0 * result["Fe2O3"]
        pd.testing.assert_series_equal(
            fe2_remaining + fe3_as_feo, total_feo, check_names=False
        )
        assert result["Fe2O3"].iloc[0] == pytest.approx(0.000876, abs=0.0001)

    def test_zero_excess(self, fe_pyroxene: pd.DataFrame) -> None:
        moles = to_moles(fe_pyroxene)
        result = oxidize_moles(moles, o_excess=0.0)
        assert "FeO" in result.columns
        assert "Fe2O3" in result.columns
        assert result["Fe2O3"].iloc[0] == pytest.approx(0.0)

    def test_large_excess_clips(self, fe_pyroxene: pd.DataFrame) -> None:
        moles = to_moles(fe_pyroxene)
        total_feo = moles["FeO"].iloc[0]
        total = moles.sum(axis=1).iloc[0]
        o_excess_max = total_feo / (2.0 * total / 100.0)
        result = oxidize_moles(moles, o_excess=o_excess_max)
        assert result["FeO"].iloc[0] == pytest.approx(0.0)
        assert result["Fe2O3"].iloc[0] == pytest.approx(total_feo / 2.0, abs=0.0001)

    def test_idempotent(self, fe_pyroxene: pd.DataFrame) -> None:
        moles = to_moles(fe_pyroxene)
        first = oxidize_moles(moles, o_excess=0.05)
        second = oxidize_moles(first, o_excess=0.05)
        pd.testing.assert_frame_equal(first, second)

    def test_multi_row_series(self, fe_pyroxene: pd.DataFrame) -> None:
        multi = pd.concat([fe_pyroxene] * 3, ignore_index=True)
        moles = to_moles(multi)
        o_excess = pd.Series([0.0, 0.5, 1.0])
        result = oxidize_moles(moles, o_excess=o_excess)
        assert result["Fe2O3"].iloc[0] == pytest.approx(0.0)
        assert result["Fe2O3"].iloc[1] > result["Fe2O3"].iloc[0]
        assert result["Fe2O3"].iloc[2] > result["Fe2O3"].iloc[1]

    def test_no_iron(self) -> None:
        df = pd.DataFrame({"SiO2": [55.0], "MgO": [18.0], "CaO": [25.0]})
        result = oxidize_moles(df, o_excess=0.05)
        pd.testing.assert_frame_equal(result, df)


class TestMolecularWeights:
    def test_series(self) -> None:
        result = molecular_weights(["SiO2", "FeO", "Al2O3"])
        assert result["SiO2"] == pytest.approx(60.084, abs=0.01)
        assert result["FeO"] == pytest.approx(71.844, abs=0.01)


# ---------------------------------------------------------------------------
# FeO ↔ Fe₂O₃ interconversion
# ---------------------------------------------------------------------------

_MW_FEO = MW("FeO")
_MW_FE2O3 = MW("Fe2O3")


class TestFeoToFe2O3:
    def test_basic(self) -> None:
        df = pd.DataFrame({"SiO2": [50.0], "FeO": [10.0], "Al2O3": [5.0]})
        result = feo_to_fe2o3(df)

        expected_fe2o3 = (10.0 / _MW_FEO) * 0.5 * _MW_FE2O3
        assert result["Fe2O3"].iloc[0] == pytest.approx(expected_fe2o3, abs=0.01)
        assert "FeO" not in result.columns
        assert "SiO2" in result.columns

    def test_merges_with_existing_fe2o3(self) -> None:
        df = pd.DataFrame({"FeO": [10.0], "Fe2O3": [5.0]})
        result = feo_to_fe2o3(df)

        fe2o3_from_feo = (10.0 / _MW_FEO) * 0.5 * _MW_FE2O3
        assert result["Fe2O3"].iloc[0] == pytest.approx(5.0 + fe2o3_from_feo, abs=0.01)
        assert "FeO" not in result.columns

    def test_no_iron(self) -> None:
        df = pd.DataFrame({"SiO2": [50.0], "Al2O3": [5.0]})
        result = feo_to_fe2o3(df)
        assert list(result.columns) == ["SiO2", "Al2O3"]

    def test_does_not_mutate(self) -> None:
        df = pd.DataFrame({"FeO": [10.0]})
        _ = feo_to_fe2o3(df)
        assert "FeO" in df.columns


class TestFe2O3ToFeO:
    def test_basic(self) -> None:
        df = pd.DataFrame({"SiO2": [50.0], "Fe2O3": [10.0], "Al2O3": [5.0]})
        result = fe2o3_to_feo(df)

        expected_feo = (10.0 / _MW_FE2O3) * 2 * _MW_FEO
        assert result["FeO"].iloc[0] == pytest.approx(expected_feo, abs=0.01)
        assert "Fe2O3" not in result.columns
        assert "SiO2" in result.columns

    def test_merges_with_existing_feo(self) -> None:
        df = pd.DataFrame({"FeO": [5.0], "Fe2O3": [10.0]})
        result = fe2o3_to_feo(df)

        feo_from_fe2o3 = (10.0 / _MW_FE2O3) * 2 * _MW_FEO
        assert result["FeO"].iloc[0] == pytest.approx(5.0 + feo_from_fe2o3, abs=0.01)
        assert "Fe2O3" not in result.columns

    def test_no_iron(self) -> None:
        df = pd.DataFrame({"SiO2": [50.0], "Al2O3": [5.0]})
        result = fe2o3_to_feo(df)
        assert list(result.columns) == ["SiO2", "Al2O3"]

    def test_does_not_mutate(self) -> None:
        df = pd.DataFrame({"Fe2O3": [10.0]})
        _ = fe2o3_to_feo(df)
        assert "Fe2O3" in df.columns


class TestReduceMoles:
    def test_basic(self) -> None:
        df = pd.DataFrame({"SiO2": [0.5], "Fe2O3": [0.1], "Al2O3": [0.05]})
        result = reduce_moles(df)

        assert result["FeO"].iloc[0] == pytest.approx(0.2, abs=1e-6)
        assert "Fe2O3" not in result.columns
        assert "SiO2" in result.columns

    def test_merges_with_existing_feo(self) -> None:
        df = pd.DataFrame({"FeO": [0.3], "Fe2O3": [0.1]})
        result = reduce_moles(df)

        assert result["FeO"].iloc[0] == pytest.approx(0.5, abs=1e-6)
        assert "Fe2O3" not in result.columns

    def test_no_iron(self) -> None:
        df = pd.DataFrame({"SiO2": [0.5], "Al2O3": [0.05]})
        result = reduce_moles(df)
        assert list(result.columns) == ["SiO2", "Al2O3"]

    def test_does_not_mutate(self) -> None:
        df = pd.DataFrame({"Fe2O3": [0.1]})
        _ = reduce_moles(df)
        assert "Fe2O3" in df.columns


# ---------------------------------------------------------------------------
# TestConvert
# ---------------------------------------------------------------------------

_MW_SIO2 = MW("SiO2")
_MW_MGO = MW("MgO")
_MW_CAO = MW("CaO")
_MW_FEO = MW("FeO")
_MW_AL2O3 = MW("Al2O3")


# ---------------------------------------------------------------------------
# TestFromApfu
# ---------------------------------------------------------------------------


class TestFromApfu:
    def test_n_oxygens(self, diopside: pd.DataFrame) -> None:
        apfu = to_apfu(diopside, n_oxygens=6)
        result = from_apfu(apfu, n_oxygens=6)
        expected = from_apfu(apfu, n_oxygens=6, n_cations=None)
        pd.testing.assert_frame_equal(result, expected)

    def test_n_cations(self, sanidine: pd.DataFrame) -> None:
        apfu = to_apfu(sanidine, n_cations=4)
        result = from_apfu(apfu, n_cations=4)
        assert "SiO2" in result.columns
        assert "Al2O3" in result.columns
        assert "K2O" in result.columns

    def test_n_cations_roundtrip(self, sanidine: pd.DataFrame) -> None:
        ox = sanidine[["SiO2", "Al2O3", "K2O"]]
        total = ox.sum(axis=1).iloc[0]
        apfu = to_apfu(ox, n_cations=4)
        result = from_apfu(apfu, n_cations=4, total=total)
        pd.testing.assert_frame_equal(result, ox)

    def test_n_oxygens_roundtrip(self, diopside: pd.DataFrame) -> None:
        total = diopside.sum(axis=1).iloc[0]
        apfu = to_apfu(diopside, n_oxygens=6)
        result = from_apfu(apfu, n_oxygens=6, total=total)
        pd.testing.assert_frame_equal(result, diopside)

    def test_exactly_one_param(self, diopside: pd.DataFrame) -> None:
        apfu = to_apfu(diopside, n_oxygens=6)
        with pytest.raises(ValueError, match="Specify exactly one"):
            from_apfu(apfu)
        with pytest.raises(ValueError, match="Specify exactly one"):
            from_apfu(apfu, n_oxygens=6, n_cations=4)

    def test_total_normalises(self, diopside: pd.DataFrame) -> None:
        apfu = to_apfu(diopside, n_oxygens=6)
        result = from_apfu(apfu, n_oxygens=6, total=100.0)
        assert result.sum(axis=1).iloc[0] == pytest.approx(100.0)

    # ---------------------------------------------------------------------------
    # TestConvert
    # ---------------------------------------------------------------------------

    def test_wt_to_moles_matches_to_moles(self, diopside: pd.DataFrame) -> None:
        result = convert(diopside, "moles")
        expected = to_moles(diopside)
        pd.testing.assert_frame_equal(result, expected)

    def test_moles_to_wt_matches_to_oxides(self, diopside: pd.DataFrame) -> None:
        moles = to_moles(diopside)
        result = convert(moles, "wt%", from_unit="moles")
        expected = to_oxides(moles)
        pd.testing.assert_frame_equal(result, expected)

    def test_wt_to_apfu_n_oxygens(self, diopside: pd.DataFrame) -> None:
        result = convert(diopside, "apfu", n_oxygens=6)
        expected = to_apfu(diopside, n_oxygens=6)
        pd.testing.assert_frame_equal(result, expected)

    def test_wt_to_apfu_n_cations(self, sanidine: pd.DataFrame) -> None:
        result = convert(sanidine, "apfu", n_cations=4)
        expected = to_apfu(sanidine, n_cations=4)
        pd.testing.assert_frame_equal(result, expected)

    def test_moles_to_apfu(self, diopside: pd.DataFrame) -> None:
        moles = to_moles(diopside)
        result = convert(moles, "apfu", from_unit="moles", n_oxygens=6)
        expected = to_apfu(moles, n_oxygens=6, units="moles")
        pd.testing.assert_frame_equal(result, expected)

    def test_apfu_to_wt(self, diopside: pd.DataFrame) -> None:
        apfu = to_apfu(diopside, n_oxygens=6)
        result = convert(apfu, "wt%", from_unit="apfu", n_oxygens=6)
        expected = from_apfu(apfu, n_oxygens=6)
        pd.testing.assert_frame_equal(result, expected)

    def test_apfu_to_moles(self, diopside: pd.DataFrame) -> None:
        apfu = to_apfu(diopside, n_oxygens=6)
        result = convert(apfu, "moles", from_unit="apfu", n_oxygens=6)
        oxide_wt = from_apfu(apfu, n_oxygens=6)
        expected = to_moles(oxide_wt)
        pd.testing.assert_frame_equal(result, expected)

    def test_same_unit_returns_copy(self, diopside: pd.DataFrame) -> None:
        result = convert(diopside, "wt%")
        pd.testing.assert_frame_equal(result, diopside)
        assert result is not diopside

    def test_from_unit_defaults_to_wt(self, diopside: pd.DataFrame) -> None:
        result = convert(diopside, "moles")
        expected = to_moles(diopside)
        pd.testing.assert_frame_equal(result, expected)

    def test_from_unit_explicit(self, diopside: pd.DataFrame) -> None:
        moles = to_moles(diopside)
        result = convert(moles, "wt%", from_unit="moles")
        expected = to_oxides(moles)
        pd.testing.assert_frame_equal(result, expected)

    def test_apfu_attrs_fallback_n_oxygens(self, diopside: pd.DataFrame) -> None:
        apfu = to_apfu(diopside, n_oxygens=6)
        apfu.attrs["petro_n_oxygens"] = 6
        result = convert(apfu, "wt%", from_unit="apfu")
        expected = from_apfu(apfu, n_oxygens=6)
        pd.testing.assert_frame_equal(result, expected)

    def test_apfu_no_params_raises(self, diopside: pd.DataFrame) -> None:
        apfu = to_apfu(diopside, n_oxygens=6)
        with pytest.raises(ValueError, match="Specify exactly one"):
            convert(apfu, "wt%", from_unit="apfu")

    def test_to_apfu_no_params_raises(self, diopside: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="n_oxygens"):
            convert(diopside, "apfu")

    def test_to_apfu_both_params_raises(self, diopside: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Specify exactly one"):
            convert(diopside, "apfu", n_oxygens=6, n_cations=4)

    def test_invalid_to_unit_raises(self, diopside: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Invalid to_unit"):
            convert(diopside, "grams")

    def test_invalid_from_unit_raises(self, diopside: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Invalid from_unit"):
            convert(diopside, "wt%", from_unit="grams")

    def test_apfu_to_wt_proportions(self, diopside: pd.DataFrame) -> None:
        """apfu → wt% without total returns proportional masses."""
        apfu = convert(diopside, "apfu", n_oxygens=6)
        result = convert(apfu, "wt%", from_unit="apfu", n_oxygens=6)
        expected = from_apfu(apfu, n_oxygens=6)
        pd.testing.assert_frame_equal(result, expected)

    def test_wt_to_moles_to_wt_roundtrip(self, diopside: pd.DataFrame) -> None:
        moles = convert(diopside, "moles")
        result = convert(moles, "wt%", from_unit="moles")
        pd.testing.assert_frame_equal(result, diopside)

    def test_apfu_to_wt_n_cations(self, sanidine: pd.DataFrame) -> None:
        apfu = convert(sanidine, "apfu", n_cations=4)
        result = convert(apfu, "wt%", from_unit="apfu", n_cations=4)
        expected = from_apfu(apfu, n_cations=4)
        pd.testing.assert_frame_equal(result, expected)

    def test_apfu_to_moles_n_cations(self, sanidine: pd.DataFrame) -> None:
        apfu = convert(sanidine, "apfu", n_cations=4)
        result = convert(apfu, "moles", from_unit="apfu", n_cations=4)
        oxide_wt = from_apfu(apfu, n_cations=4)
        expected = to_moles(oxide_wt)
        pd.testing.assert_frame_equal(result, expected)

    def test_apfu_attrs_fallback_n_cations(self, sanidine: pd.DataFrame) -> None:
        apfu = convert(sanidine, "apfu", n_cations=4)
        apfu.attrs["petro_n_cations"] = 4
        result = convert(apfu, "wt%", from_unit="apfu")
        expected = from_apfu(apfu, n_cations=4)
        pd.testing.assert_frame_equal(result, expected)

    def test_apfu_both_params_raises(self, diopside: pd.DataFrame) -> None:
        apfu = convert(diopside, "apfu", n_oxygens=6)
        with pytest.raises(ValueError, match="Specify exactly one"):
            convert(apfu, "wt%", from_unit="apfu", n_oxygens=6, n_cations=4)
