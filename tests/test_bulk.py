"""Tests for BulkAccessor and bulk-rock calculations."""

from __future__ import annotations

import pandas as pd
import pytest

import petropandas  # noqa: F401 — triggers accessor registration
from petropandas._calc import (
    alumina_saturation,
    apatite_correction,
    cipw_norm,
    oxide_ratios,
)


# ---------------------------------------------------------------------------
# BulkAccessor basics
# ---------------------------------------------------------------------------


class TestBulkAccessor:
    def test_call_returns_copy(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk()
        assert result is not granite_bulk

    def test_call_idempotent(self, granite_bulk: pd.DataFrame) -> None:
        first = granite_bulk.bulk()
        second = first.bulk()
        pd.testing.assert_frame_equal(first, second)

    def test_call_cleans_aliases(self) -> None:
        df = pd.DataFrame({"SiO2": [70.0], "FeO*": [3.0], "MgO": [1.0]})
        result = df.bulk()
        assert "FeO" in result.columns

    def test_returns_all_columns(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk()
        assert list(result.columns) == list(granite_bulk.columns)


# ---------------------------------------------------------------------------
# CIPW norm
# ---------------------------------------------------------------------------


class TestCipwNorm:
    def test_granite_has_quartz(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.cipw()
        assert "Qz" in result.columns
        assert result["Qz"].iloc[0] > 0

    def test_granite_has_feldspars(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.cipw()
        assert "Or" in result.columns
        assert "Ab" in result.columns
        assert "An" in result.columns

    def test_basalt_has_diopside(self, basalt_bulk: pd.DataFrame) -> None:
        result = basalt_bulk.bulk.cipw()
        assert "Di" in result.columns
        assert result["Di"].iloc[0] > 0

    def test_basalt_has_olivine_or_hypersthene(self, basalt_bulk: pd.DataFrame) -> None:
        result = basalt_bulk.bulk.cipw()
        has_hy = "Hy" in result.columns and result["Hy"].iloc[0] > 0
        has_ol = "Ol" in result.columns and result["Ol"].iloc[0] > 0
        assert has_hy or has_ol

    def test_sums_approximately_100(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.cipw()
        total = result.sum(axis=1).iloc[0]
        assert total == pytest.approx(100.0, abs=2.0)

    def test_basalt_sums_approximately_100(self, basalt_bulk: pd.DataFrame) -> None:
        result = basalt_bulk.bulk.cipw()
        total = result.sum(axis=1).iloc[0]
        assert total == pytest.approx(100.0, abs=2.0)

    def test_minerals_non_negative(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.cipw()
        assert (result >= 0).all().all()

    def test_has_iron_oxides(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.cipw()
        assert "Il" in result.columns or "Mt" in result.columns

    def test_diorite_norm(self, diorite_bulk: pd.DataFrame) -> None:
        result = diorite_bulk.bulk.cipw()
        total = result.sum(axis=1).iloc[0]
        assert total == pytest.approx(100.0, abs=2.0)

    def test_all_output_columns_non_empty(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.cipw()
        for col in result.columns:
            assert result[col].notna().all(), f"NaN in column {col}"


# ---------------------------------------------------------------------------
# Alumina saturation
# ---------------------------------------------------------------------------


class TestAluminaSaturation:
    def test_output_columns(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.alumina_saturation()
        assert "A/NK" in result.columns
        assert "A/CNK" in result.columns

    def test_granite_peraluminous(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.alumina_saturation()
        assert result["A/CNK"].iloc[0] > 1.0

    def test_basalt_metaluminous(self, basalt_bulk: pd.DataFrame) -> None:
        result = basalt_bulk.bulk.alumina_saturation()
        assert result["A/CNK"].iloc[0] < 1.0
        assert result["A/NK"].iloc[0] >= 1.0

    def test_classify_column(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.alumina_saturation(classify=True)
        assert "shand_class" in result.columns
        assert result["shand_class"].iloc[0] == "peraluminous"

    def test_classify_basalt(self, basalt_bulk: pd.DataFrame) -> None:
        result = basalt_bulk.bulk.alumina_saturation(classify=True)
        assert result["shand_class"].iloc[0] == "metaluminous"

    def test_no_classify_by_default(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.alumina_saturation()
        assert "shand_class" not in result.columns

    def test_a_nk_ratio(self, basalt_bulk: pd.DataFrame) -> None:
        result = basalt_bulk.bulk.alumina_saturation()
        assert result["A/NK"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Oxide ratios
# ---------------------------------------------------------------------------


class TestOxideRatios:
    def test_output_columns(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.oxide_ratios()
        assert "FeOT" in result.columns
        assert "Mg#" in result.columns
        assert "Na2O+K2O" in result.columns
        assert "K2O/Na2O" in result.columns

    def test_feot_calculation(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.oxide_ratios()
        expected_feot = 1.8 + 0.8998 * 1.2
        assert result["FeOT"].iloc[0] == pytest.approx(expected_feot, abs=0.01)

    def test_mg_number(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.oxide_ratios()
        assert 0.0 < result["Mg#"].iloc[0] < 1.0

    def test_total_alkalis(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.oxide_ratios()
        assert result["Na2O+K2O"].iloc[0] == pytest.approx(7.7, abs=0.01)

    def test_k_to_na_ratio(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.oxide_ratios()
        assert result["K2O/Na2O"].iloc[0] == pytest.approx(4.5 / 3.2, abs=0.01)

    def test_pass_through_sio2(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.oxide_ratios()
        assert "SiO2" in result.columns
        assert result["SiO2"].iloc[0] == pytest.approx(72.0)

    def test_missing_oxide_ratio_is_nan(self) -> None:
        df = pd.DataFrame({"SiO2": [70.0], "MgO": [2.0]})
        result = df.bulk.oxide_ratios()
        assert "Mg#" not in result.columns
        assert "FeOT" not in result.columns


# ---------------------------------------------------------------------------
# Direct _calc functions
# ---------------------------------------------------------------------------


class TestCalcAluminaSaturation:
    def test_matches_accessor(self, granite_bulk: pd.DataFrame) -> None:
        from_accessor = granite_bulk.bulk.alumina_saturation()
        from_func = alumina_saturation(granite_bulk)
        pd.testing.assert_frame_equal(from_accessor, from_func)


class TestCalcOxideRatios:
    def test_matches_accessor(self, granite_bulk: pd.DataFrame) -> None:
        from_accessor = granite_bulk.bulk.oxide_ratios()
        from_func = oxide_ratios(granite_bulk)
        pd.testing.assert_frame_equal(from_accessor, from_func)


class TestCalcCipwNorm:
    def test_matches_accessor(self, granite_bulk: pd.DataFrame) -> None:
        from_accessor = granite_bulk.bulk.cipw()
        from_func = cipw_norm(granite_bulk)
        pd.testing.assert_frame_equal(from_accessor, from_func)


# ---------------------------------------------------------------------------
# Apatite correction
# ---------------------------------------------------------------------------


class TestApatiteCorrection:
    def test_granite_reduces_cao(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.apatite_correction()
        # P2O5 = 0.12 wt% → CaO consumed ≈ 0.157 wt%
        assert result["CaO"].iloc[0] < granite_bulk["CaO"].iloc[0]

    def test_granite_p2o5_zeroed(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.apatite_correction()
        assert (result["P2O5"] == 0.0).all()

    def test_basalt_reduces_cao(self, basalt_bulk: pd.DataFrame) -> None:
        result = basalt_bulk.bulk.apatite_correction()
        # P2O5 = 0.25 → CaO consumed ≈ 0.327 wt%
        assert result["CaO"].iloc[0] < basalt_bulk["CaO"].iloc[0]

    def test_basalt_p2o5_zeroed(self, basalt_bulk: pd.DataFrame) -> None:
        result = basalt_bulk.bulk.apatite_correction()
        assert (result["P2O5"] == 0.0).all()

    def test_zero_p2o5_unchanged(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [72.0], "Al2O3": [14.0], "CaO": [1.8], "P2O5": [0.0]}
        )
        result = df.bulk.apatite_correction()
        pd.testing.assert_frame_equal(result, df)

    def test_no_p2o5_column_unchanged(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0], "CaO": [1.8]})
        result = df.bulk.apatite_correction()
        pd.testing.assert_frame_equal(result, df)

    def test_no_cao_column_zeros_p2o5(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "P2O5": [0.25]})
        result = df.bulk.apatite_correction()
        assert (result["P2O5"] == 0.0).all()

    def test_columns_preserved(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.apatite_correction()
        assert list(result.columns) == list(granite_bulk.columns)

    def test_does_not_mutate(self, granite_bulk: pd.DataFrame) -> None:
        original_cao = granite_bulk["CaO"].iloc[0]
        granite_bulk.bulk.apatite_correction()
        assert granite_bulk["CaO"].iloc[0] == original_cao

    def test_correction_amount(self) -> None:
        """Verify exact correction for a known composition."""
        from petropandas._core import MW

        df = pd.DataFrame({"SiO2": [72.0], "CaO": [2.0], "P2O5": [1.0]})
        result = df.bulk.apatite_correction()

        expected_cao = 2.0 - (10.0 / 3.0) * (1.0 / MW("P2O5")) * MW("CaO")
        assert result["CaO"].iloc[0] == pytest.approx(expected_cao, rel=1e-6)


class TestCalcApatiteCorrection:
    def test_matches_accessor(self, granite_bulk: pd.DataFrame) -> None:
        from_accessor = granite_bulk.bulk.apatite_correction()
        from_func = apatite_correction(granite_bulk)
        pd.testing.assert_frame_equal(from_accessor, from_func)


# ---------------------------------------------------------------------------
# THERMOCALC bulk
# ---------------------------------------------------------------------------


class TestTCbulk:
    def test_granite_prints_bulk(self, granite_bulk: pd.DataFrame, capsys) -> None:
        granite_bulk.bulk.TCbulk()
        out = capsys.readouterr().out
        assert out.startswith("bulk")
        assert "SiO2" in out

    def test_basalt_prints_bulk(self, basalt_bulk: pd.DataFrame, capsys) -> None:
        basalt_bulk.bulk.TCbulk()
        out = capsys.readouterr().out
        assert out.startswith("bulk")

    def test_dataframe_returns_df(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.TCbulk(dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert "SiO2" in result.columns
        assert "O" in result.columns

    def test_default_system_columns(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.TCbulk(system="MnNCKFMASHTO", dataframe=True)
        expected = [
            "H2O",
            "SiO2",
            "Al2O3",
            "CaO",
            "MgO",
            "FeO",
            "K2O",
            "Na2O",
            "TiO2",
            "MnO",
            "O",
        ]
        assert list(result.columns) == expected

    def test_kfmash_system(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.TCbulk(system="KFMASH", dataframe=True)
        assert list(result.columns) == ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"]

    def test_invalid_system_raises(self, granite_bulk: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Invalid system"):
            granite_bulk.bulk.TCbulk(system="INVALID")

    def test_h2o_auto_calc(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0], "CaO": [1.8], "FeO": [2.0]})
        result = df.bulk.TCbulk(system="NCKFMASHTO", dataframe=True)
        assert "H2O" in result.columns
        assert result["H2O"].iloc[0] > 0

    def test_h2o_explicit(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0], "CaO": [1.8], "FeO": [2.0]})
        result = df.bulk.TCbulk(system="NCKFMASHTO", H2O=5.0, dataframe=True)
        assert result["H2O"].iloc[0] > 0

    def test_p2o5_zeroed(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.TCbulk(dataframe=True)
        assert "P2O5" not in result.columns

    def test_fe_reduced(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [72.0], "Al2O3": [14.0], "Fe2O3": [3.0], "CaO": [1.0]}
        )
        result = df.bulk.TCbulk(dataframe=True)
        assert "Fe2O3" not in result.columns
        assert "FeO" in result.columns

    def test_missing_columns_fill_zero(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0]})
        result = df.bulk.TCbulk(system="KFMASH", dataframe=True)
        assert "MgO" in result.columns
        assert "FeO" in result.columns


# ---------------------------------------------------------------------------
# PerpleX bulk
# ---------------------------------------------------------------------------


class TestPerplexbulk:
    def test_granite_prints_component_list(
        self, granite_bulk: pd.DataFrame, capsys
    ) -> None:
        granite_bulk.bulk.Perplexbulk()
        out = capsys.readouterr().out
        assert "begin thermodynamic component list" in out
        assert "end thermodynamic component list" in out

    def test_dataframe_returns_df(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.Perplexbulk(dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert "SiO2" in result.columns

    def test_o2_column(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.Perplexbulk(dataframe=True)
        assert "O2" in result.columns
        assert "O" not in result.columns

    def test_oxygen_mult_2(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.Perplexbulk(oxygen=0.01, dataframe=True)
        assert result["O2"].iloc[0] == pytest.approx(0.02, abs=1e-6)

    def test_invalid_system_raises(self, granite_bulk: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Invalid system"):
            granite_bulk.bulk.Perplexbulk(system="INVALID")

    def test_h2o_auto_calc(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0], "CaO": [1.8], "FeO": [2.0]})
        result = df.bulk.Perplexbulk(system="NCKFMASHTO", dataframe=True)
        assert "H2O" in result.columns
        assert result["H2O"].iloc[0] > 0

    def test_kfmash_system(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.Perplexbulk(system="KFMASH", dataframe=True)
        assert list(result.columns) == ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"]


# ---------------------------------------------------------------------------
# MAGEMin bulk
# ---------------------------------------------------------------------------


class TestMAGEMin:
    def test_granite_prints_header(self, granite_bulk: pd.DataFrame, capsys) -> None:
        granite_bulk.bulk.MAGEMin()
        out = capsys.readouterr().out
        assert "# HEADER" in out
        assert "# BULK-ROCK COMPOSITION" in out

    def test_dataframe_returns_df(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.MAGEMin(dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert "SiO2" in result.columns

    def test_default_db_columns(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.MAGEMin(db="mp", dataframe=True)
        expected = [
            "SiO2",
            "Al2O3",
            "CaO",
            "MgO",
            "FeO",
            "K2O",
            "Na2O",
            "TiO2",
            "O",
            "MnO",
            "H2O",
        ]
        assert list(result.columns) == expected

    def test_ig_db_columns(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.MAGEMin(db="ig", dataframe=True)
        expected = [
            "SiO2",
            "Al2O3",
            "CaO",
            "MgO",
            "FeO",
            "K2O",
            "Na2O",
            "TiO2",
            "O",
            "Cr2O3",
            "H2O",
        ]
        assert list(result.columns) == expected

    def test_invalid_db_raises(self, granite_bulk: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Invalid database"):
            granite_bulk.bulk.MAGEMin(db="INVALID")

    def test_sys_in_wt(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.MAGEMin(db="mp", sys_in="wt", dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert "SiO2" in result.columns

    def test_title_in_output(self, granite_bulk: pd.DataFrame, capsys) -> None:
        granite_bulk.bulk.MAGEMin(title="MY_SAMPLE")
        out = capsys.readouterr().out
        assert "MY_SAMPLE" in out

    def test_comment_in_output(self, granite_bulk: pd.DataFrame, capsys) -> None:
        granite_bulk.bulk.MAGEMin(comment="test_comment")
        out = capsys.readouterr().out
        assert "test_comment" in out

    def test_h2o_auto_calc(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0], "CaO": [1.8], "FeO": [2.0]})
        result = df.bulk.MAGEMin(db="mb", dataframe=True)
        assert "H2O" in result.columns
        assert result["H2O"].iloc[0] > 0

    def test_mtl_db_no_h2o(self, granite_bulk: pd.DataFrame) -> None:
        result = granite_bulk.bulk.MAGEMin(db="mtl", dataframe=True)
        assert "H2O" not in result.columns
        assert "Na2O" in result.columns


# ---------------------------------------------------------------------------
# Shared prep behaviour
# ---------------------------------------------------------------------------


class TestThermoBulkPrep:
    def test_fe_reduced(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [72.0], "Al2O3": [14.0], "Fe2O3": [3.0], "CaO": [1.0]}
        )
        result = df.bulk.TCbulk(dataframe=True)
        assert "Fe2O3" not in result.columns
        assert "FeO" in result.columns

    def test_apatite_correction_applied(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "CaO": [2.0], "P2O5": [1.0], "FeO": [2.0]})
        result = df.bulk.TCbulk(dataframe=True)
        assert (result.get("P2O5", pd.Series([0.0])) == 0.0).all()

    def test_missing_columns_fill_zero(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0]})
        result = df.bulk.TCbulk(system="KFMASH", dataframe=True)
        for col in ["MgO", "FeO", "K2O"]:
            assert col in result.columns
            assert result[col].iloc[0] == 0.0

    def test_column_order_matches_system(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0], "FeO": [2.0], "CaO": [1.0]})
        result = df.bulk.TCbulk(system="KFMASH", dataframe=True)
        assert list(result.columns) == ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"]


# ---------------------------------------------------------------------------
# BulkAccessor.mean
# ---------------------------------------------------------------------------


class TestBulkMean:
    def test_simple_mean(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0, 70.0, 80.0], "Al2O3": [15.0, 13.0, 11.0]})
        result = df.bulk.mean()
        assert len(result) == 1
        assert result["SiO2"].iloc[0] == pytest.approx(70.0)
        assert result["Al2O3"].iloc[0] == pytest.approx(13.0)

    def test_single_row(self) -> None:
        df = pd.DataFrame({"SiO2": [72.0], "Al2O3": [14.0], "FeO": [2.0]})
        result = df.bulk.mean()
        assert result["SiO2"].iloc[0] == pytest.approx(72.0)

    def test_attrs_preserved(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0, 70.0], "Al2O3": [15.0, 13.0]})
        result = df.bulk.mean()
        assert result.attrs.get("petro_units") == "wt%"

    def test_weighted_mean(self) -> None:
        df = pd.DataFrame(
            {
                "SiO2": [60.0, 70.0, 80.0],
                "Al2O3": [15.0, 13.0, 11.0],
                "wt": [1.0, 2.0, 3.0],
            }
        )
        result = df.bulk.mean(weights="wt")
        assert result["SiO2"].iloc[0] == pytest.approx(73.333333)
        assert result["Al2O3"].iloc[0] == pytest.approx(12.333333)

    def test_weighted_excludes_weight_col(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0, 70.0], "wt": [1.0, 2.0]})
        result = df.bulk.mean(weights="wt")
        assert "wt" not in result.columns

    def test_groupby_mean(self) -> None:
        df = pd.DataFrame(
            {
                "SiO2": [60.0, 70.0, 80.0],
                "Al2O3": [15.0, 13.0, 11.0],
                "rock": ["A", "A", "B"],
            }
        )
        result = df.bulk.mean(groupby="rock")
        assert len(result) == 2
        assert result.loc["A", "SiO2"] == pytest.approx(65.0)
        assert result.loc["B", "SiO2"] == pytest.approx(80.0)

    def test_groupby_weighted_mean(self) -> None:
        df = pd.DataFrame(
            {
                "SiO2": [60.0, 70.0, 80.0, 50.0],
                "Al2O3": [15.0, 13.0, 11.0, 10.0],
                "wt": [1.0, 2.0, 3.0, 1.0],
                "rock": ["A", "A", "B", "B"],
            }
        )
        result = df.bulk.mean(groupby="rock", weights="wt")
        assert result.loc["A", "SiO2"] == pytest.approx(200.0 / 3.0)
        assert result.loc["A", "Al2O3"] == pytest.approx(41.0 / 3.0)
        assert result.loc["B", "SiO2"] == pytest.approx(72.5)
        assert result.loc["B", "Al2O3"] == pytest.approx(10.75)

    def test_groupby_index(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0, 70.0], "rock": ["X", "Y"]})
        result = df.bulk.mean(groupby="rock")
        assert list(result.index) == ["X", "Y"]

    def test_missing_weights_column_raises(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0, 70.0]})
        with pytest.raises(ValueError, match="not found"):
            df.bulk.mean(weights="MISSING")

    def test_missing_groupby_column_raises(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0, 70.0]})
        with pytest.raises(ValueError, match="not found"):
            df.bulk.mean(groupby="MISSING")

    def test_weights_zero_handled(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0, 70.0], "wt": [0.0, 2.0]})
        result = df.bulk.mean(weights="wt")
        assert result["SiO2"].iloc[0] == pytest.approx(70.0)
