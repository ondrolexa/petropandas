"""Tests for OxidesAccessor (df.oxides)."""

from __future__ import annotations

import pandas as pd
import pytest

import petropandas  # noqa: F401


class TestCallable:
    def test_returns_oxide_columns_only(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides()
        assert "SiO2" in result.columns
        assert "FeO" in result.columns
        df = pd.DataFrame({"SiO2": [50.0], "FeO": [10.0], "label": ["a"]})
        result = df.oxides()
        assert list(result.columns) == ["SiO2", "FeO"]

    def test_returns_copy(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides()
        result["SiO2"] = 0.0
        assert fe_pyroxene["SiO2"].iloc[0] == 52.00

    def test_no_oxide_columns(self) -> None:
        df = pd.DataFrame({"label": ["a"], "value": [1]})
        result = df.oxides()
        assert result.empty

    def test_from_moles(self, diopside: pd.DataFrame) -> None:
        m = diopside.moles()
        assert m.attrs.get("petro_units") == "moles"
        back = m.oxides()
        assert back.attrs.get("petro_units") == "wt%"
        pd.testing.assert_frame_equal(back, diopside[["SiO2", "MgO", "CaO"]])

    def test_idempotent(self, diopside: pd.DataFrame) -> None:
        ox1 = diopside.oxides()
        ox2 = ox1.oxides()
        pd.testing.assert_frame_equal(ox1, ox2)

    def test_from_moles_idempotent(self, diopside: pd.DataFrame) -> None:
        ox = diopside.moles().oxides()
        ox2 = ox.oxides()
        pd.testing.assert_frame_equal(ox, ox2)


class TestSorted:
    def test_major_oxides_order(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.sorted()
        expected = ["SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O"]
        assert list(result.columns) == expected

    def test_volatiles_at_end(self) -> None:
        df = pd.DataFrame(
            {
                "CO2": [1.0],
                "SiO2": [50.0],
                "H2O": [2.0],
                "FeO": [10.0],
                "SO3": [0.5],
            }
        )
        result = df.oxides.sorted()
        assert list(result.columns) == ["SiO2", "FeO", "H2O", "CO2", "SO3"]

    def test_other_oxides_alphabetical(self, fe_garnet_multi: pd.DataFrame) -> None:
        result = fe_garnet_multi.oxides.sorted()
        expected = ["SiO2", "Al2O3", "FeO", "MnO", "MgO", "CaO", "Cr2O3"]
        assert list(result.columns) == expected

    def test_non_oxide_columns_excluded(self) -> None:
        df = pd.DataFrame(
            {
                "label": ["a"],
                "FeO": [10.0],
                "SiO2": [50.0],
                "spot": [1],
            }
        )
        result = df.oxides.sorted()
        assert list(result.columns) == ["SiO2", "FeO"]

    def test_returns_copy(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.sorted()
        result["SiO2"] = 0.0
        assert fe_pyroxene["SiO2"].iloc[0] == 52.00

    def test_sorted_idempotent(self, fe_pyroxene: pd.DataFrame) -> None:
        s1 = fe_pyroxene.oxides.sorted()
        s2 = s1.oxides.sorted()
        pd.testing.assert_frame_equal(s1, s2)


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------


class TestMean:
    def test_single_row_unchanged(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.mean()
        assert len(result) == 1
        assert result["SiO2"].iloc[0] == pytest.approx(52.0)

    def test_multi_row(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0, 70.0], "Al2O3": [15.0, 13.0], "FeO": [5.0, 3.0]}
        )
        result = df.oxides.mean()
        assert len(result) == 1
        assert result["SiO2"].iloc[0] == pytest.approx(65.0)
        assert result["Al2O3"].iloc[0] == pytest.approx(14.0)
        assert result["FeO"].iloc[0] == pytest.approx(4.0)

    def test_columns_match(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.mean()
        assert list(result.columns) == list(fe_pyroxene.oxides().columns)

    def test_attrs_wt(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.mean()
        assert result.attrs.get("petro_units") == "wt%"

    def test_groupby(self) -> None:
        df = pd.DataFrame(
            {
                "SiO2": [60.0, 70.0, 80.0],
                "Al2O3": [15.0, 13.0, 11.0],
                "FeO": [5.0, 3.0, 1.0],
                "sample": ["A", "A", "B"],
            }
        )
        result = df.oxides.mean(groupby="sample")
        assert len(result) == 2
        assert result.loc["A", "SiO2"] == pytest.approx(65.0)
        assert result.loc["B", "SiO2"] == pytest.approx(80.0)

    def test_groupby_index(self) -> None:
        df = pd.DataFrame(
            {
                "SiO2": [60.0, 70.0],
                "Al2O3": [15.0, 13.0],
                "sample": ["X", "Y"],
            }
        )
        result = df.oxides.mean(groupby="sample")
        assert list(result.index) == ["X", "Y"]

    def test_groupby_attrs(self) -> None:
        df = pd.DataFrame(
            {
                "SiO2": [60.0, 70.0],
                "Al2O3": [15.0, 13.0],
                "sample": ["A", "B"],
            }
        )
        result = df.oxides.mean(groupby="sample")
        assert result.attrs.get("petro_units") == "wt%"

    def test_groupby_missing_column_raises(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0], "Al2O3": [15.0]})
        with pytest.raises(ValueError, match="not found"):
            df.oxides.mean(groupby="missing")


# ---------------------------------------------------------------------------
# TestSelect
# ---------------------------------------------------------------------------


class TestSelect:
    def test_str_on_index_contains(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0, 70.0], "FeO": [8.0, 9.0]},
            index=["sample_1", "sample_2"],
        )
        result = df.oxides.select("sample_1")
        assert len(result) == 1
        assert result.index[0] == "sample_1"

    def test_str_on_column_contains(self) -> None:
        df = pd.DataFrame(
            {
                "oxide": ["SiO2", "FeO", "Fe2O3", "MgO"],
                "value": [60.0, 8.0, 2.0, 10.0],
            }
        )
        result = df.oxides.select("Fe", on="oxide")
        assert list(result["oxide"]) == ["FeO", "Fe2O3"]

    def test_str_case_sensitive(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0], "FeO": [8.0]},
            index=["FeO_point", "SiO2_point"],
        )
        result = df.oxides.select("fe")
        assert len(result) == 0

    def test_str_no_match_returns_empty(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0], "FeO": [8.0]},
            index=["a", "b"],
        )
        result = df.oxides.select("Zzzz")
        assert len(result) == 0
        assert list(result.columns) == ["SiO2", "FeO"]

    def test_str_numeric_index_warning(self) -> None:
        import warnings

        df = pd.DataFrame({"SiO2": [60.0], "FeO": [8.0]}, index=[0, 1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df.oxides.select("0")
            assert len(w) == 1
            assert "numeric" in str(w[0].message).lower()

    def test_list_on_index_exact(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0, 70.0, 80.0]},
            index=["a", "b", "c"],
        )
        result = df.oxides.select(["a", "c"])
        assert len(result) == 2
        assert list(result.index) == ["a", "c"]

    def test_list_on_column_exact(self) -> None:
        df = pd.DataFrame(
            {
                "oxide": ["SiO2", "FeO", "Fe2O3", "MgO"],
                "value": [60.0, 8.0, 2.0, 10.0],
            }
        )
        result = df.oxides.select(["FeO", "MgO"], on="oxide")
        assert list(result["oxide"]) == ["FeO", "MgO"]

    def test_list_no_numeric_warning(self) -> None:
        import warnings

        df = pd.DataFrame({"SiO2": [60.0], "FeO": [8.0]}, index=[0, 1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = df.oxides.select([0])
            assert len(result) == 1
            assert len(w) == 0

    def test_list_empty_returns_empty(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0], "FeO": [8.0]},
            index=["a", "b"],
        )
        result = df.oxides.select([])
        assert len(result) == 0

    def test_list_partial_index_match(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0, 70.0, 80.0]},
            index=[1, 10, 11],
        )
        result = df.oxides.select([1, 10])
        assert list(result.index) == [1, 10]

    def test_bool_series(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0, 70.0, 80.0], "FeO": [8.0, 9.0, 10.0]},
            index=["a", "b", "c"],
        )
        mask = pd.Series([True, False, True], index=["a", "b", "c"])
        result = df.oxides.select(mask)
        assert list(result.index) == ["a", "c"]

    def test_bool_series_requires_same_index(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0, 70.0], "FeO": [8.0, 9.0]},
            index=["a", "b"],
        )
        mask = pd.Series([True, False], index=["x", "y"])
        with pytest.raises(ValueError, match="does not match"):
            df.oxides.select(mask)

    def test_bool_on_ignored(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0, 70.0], "FeO": [8.0, 9.0]},
            index=["a", "b"],
        )
        mask = pd.Series([True, False], index=["a", "b"])
        result = df.oxides.select(mask, on="SiO2")
        assert len(result) == 1
        assert result.index[0] == "a"

    def test_attrs_preserved(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0], "FeO": [8.0]},
            index=["a"],
        )
        df.attrs["petro_units"] = "moles"
        result = df.oxides.select(["a"])
        assert result.attrs.get("petro_units") == "moles"

    def test_returns_copy(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [60.0, 70.0], "FeO": [8.0, 9.0]},
            index=["a", "b"],
        )
        result = df.oxides.select(["a"])
        result.iloc[0, 0] = 999.0
        assert df.iloc[0, 0] == 60.0

    def test_invalid_arg_type_raises(self) -> None:
        df = pd.DataFrame({"SiO2": [60.0], "FeO": [8.0]})
        with pytest.raises(TypeError, match="arg must be"):
            df.oxides.select(42)  # type: ignore[arg-type]
