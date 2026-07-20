"""Tests for _core.py — oxide and ion utilities."""

from __future__ import annotations

import pytest

from petropandas._core import (
    ALIASES,
    MW,
    _cations_per,
    _detect_col,
    _detect_cols,
    _element_of,
    _formula_cols,
    _ion_name,
    _is_formula,
    _is_oxide,
    _oxide_cols,
    _oxygens_per,
    _parse_ion,
)

import pandas as pd


class TestIsOxide:
    def test_sio2(self) -> None:
        assert _is_oxide("SiO2")

    def test_feo(self) -> None:
        assert _is_oxide("FeO")

    def test_al2o3(self) -> None:
        assert _is_oxide("Al2O3")

    def test_label(self) -> None:
        assert not _is_oxide("label")

    def test_total(self) -> None:
        assert not _is_oxide("total")

    def test_feo_star(self) -> None:
        assert not _is_oxide("FeO*")


class TestOxideCols:
    def test_mixed(self) -> None:
        df = pd.DataFrame({"SiO2": [1], "label": ["a"], "FeO": [2]})
        assert _oxide_cols(df) == ["SiO2", "FeO"]


class TestElementOf:
    def test_sio2(self) -> None:
        assert _element_of("SiO2") == "Si"

    def test_fe2o3(self) -> None:
        assert _element_of("Fe2O3") == "Fe"

    def test_al2o3(self) -> None:
        assert _element_of("Al2O3") == "Al"

    def test_nao(self) -> None:
        assert _element_of("Na2O") == "Na"


class TestCationsPerOxygensPer:
    def test_sio2(self) -> None:
        assert _cations_per("SiO2") == 1
        assert _oxygens_per("SiO2") == 2

    def test_al2o3(self) -> None:
        assert _cations_per("Al2O3") == 2
        assert _oxygens_per("Al2O3") == 3

    def test_feo(self) -> None:
        assert _cations_per("FeO") == 1
        assert _oxygens_per("FeO") == 1


class TestMW:
    def test_sio2(self) -> None:
        assert MW("SiO2") == pytest.approx(60.084, abs=0.01)

    def test_feo(self) -> None:
        assert MW("FeO") == pytest.approx(71.844, abs=0.01)

    def test_al2o3(self) -> None:
        assert MW("Al2O3") == pytest.approx(101.960, abs=0.01)


class TestIonName:
    def test_fe2(self) -> None:
        assert _ion_name("Fe", 2) == "Fe{2+}"

    def test_fe3(self) -> None:
        assert _ion_name("Fe", 3) == "Fe{3+}"

    def test_si4(self) -> None:
        assert _ion_name("Si", 4) == "Si{4+}"

    def test_na1(self) -> None:
        assert _ion_name("Na", 1) == "Na{+}"

    def test_o2_minus(self) -> None:
        assert _ion_name("O", -2) == "O{2-}"


class TestParseIon:
    def test_fe2(self) -> None:
        assert _parse_ion("Fe{2+}") == ("Fe", 2)

    def test_fe3(self) -> None:
        assert _parse_ion("Fe{3+}") == ("Fe", 3)

    def test_si4(self) -> None:
        assert _parse_ion("Si{4+}") == ("Si", 4)

    def test_na(self) -> None:
        assert _parse_ion("Na{+}") == ("Na", 1)

    def test_not_ion(self) -> None:
        assert _parse_ion("FeO") is None

    def test_not_parseable(self) -> None:
        assert _parse_ion("label") is None


class TestDetectCol:
    def test_oxide(self) -> None:
        df = pd.DataFrame({"SiO2": [1], "FeO": [2], "MgO": [3]})
        assert _detect_col(df, "Fe") == "FeO"

    def test_ion(self) -> None:
        df = pd.DataFrame({"Si{4+}": [1], "Fe{2+}": [2], "Mg{2+}": [3]})
        assert _detect_col(df, "Fe") == "Fe{2+}"

    def test_not_found(self) -> None:
        df = pd.DataFrame({"SiO2": [1], "MgO": [2]})
        with pytest.raises(KeyError, match="No column found"):
            _detect_col(df, "Fe")


class TestDetectCols:
    def test_multiple_fe(self) -> None:
        df = pd.DataFrame({"Fe{2+}": [1], "Fe{3+}": [2], "Si{4+}": [3]})
        result = _detect_cols(df, "Fe")
        assert result == ["Fe{2+}", "Fe{3+}"]

    def test_single(self) -> None:
        df = pd.DataFrame({"FeO": [1], "SiO2": [2]})
        result = _detect_cols(df, "Fe")
        assert result == ["FeO"]


class TestAliases:
    def test_feo_star(self) -> None:
        assert ALIASES["FeO*"] == "FeO"

    def test_feot(self) -> None:
        assert ALIASES["FeOT"] == "FeO"


class TestIsFormula:
    def test_oxide(self) -> None:
        assert _is_formula("SiO2")

    def test_element_f(self) -> None:
        assert _is_formula("F")

    def test_element_s(self) -> None:
        assert _is_formula("S")

    def test_element_cl(self) -> None:
        assert _is_formula("Cl")

    def test_element_ba(self) -> None:
        assert _is_formula("Ba")

    def test_label(self) -> None:
        assert not _is_formula("label")

    def test_total(self) -> None:
        assert not _is_formula("Total")

    def test_feo_star(self) -> None:
        assert not _is_formula("FeO*")


class TestFormulaCols:
    def test_mixed(self) -> None:
        df = pd.DataFrame(
            {"SiO2": [1], "label": ["a"], "FeO": [2], "F": [0.5], "Cl": [0.1]}
        )
        assert _formula_cols(df) == ["SiO2", "FeO", "F", "Cl"]

    def test_oxides_only(self) -> None:
        df = pd.DataFrame({"SiO2": [1], "FeO": [2]})
        assert _formula_cols(df) == ["SiO2", "FeO"]

    def test_no_formulas(self) -> None:
        df = pd.DataFrame({"Sample": ["a"], "Locality": ["b"]})
        assert _formula_cols(df) == []
