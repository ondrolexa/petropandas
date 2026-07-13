"""Tests for PetroSeriesAccessor (series.petro)."""

from __future__ import annotations

import pandas as pd
import pytest

import petropandas  # noqa: F401


class TestIsOxide:
    def test_known_oxide(self) -> None:
        s = pd.Series([50.0], name="SiO2")
        assert s.mineral.is_oxide is True

    def test_unknown_column(self) -> None:
        s = pd.Series([50.0], name="label")
        assert s.mineral.is_oxide is False


class TestElement:
    def test_sio2(self) -> None:
        s = pd.Series([50.0], name="SiO2")
        assert s.mineral.element == "Si"

    def test_feo(self) -> None:
        s = pd.Series([10.0], name="FeO")
        assert s.mineral.element == "Fe"

    def test_unknown(self) -> None:
        s = pd.Series([10.0], name="total")
        assert s.mineral.element is None


class TestMolecularWeight:
    def test_sio2(self) -> None:
        s = pd.Series([50.0], name="SiO2")
        assert s.mineral.molecular_weight == pytest.approx(60.084, abs=0.01)

    def test_unknown(self) -> None:
        s = pd.Series([10.0], name="total")
        assert s.mineral.molecular_weight is None


class TestToMole:
    def test_sio2(self) -> None:
        s = pd.Series([55.49], name="SiO2")
        result = s.mineral.to_mole()
        assert result.iloc[0] == pytest.approx(0.9235, abs=0.001)


class TestToCation:
    def test_sio2_diopside(self) -> None:
        s = pd.Series([55.49], name="SiO2")
        # For diopside: n_oxygens=6, total_oxygens=2.7706
        # APFU = (55.49/60.084) * 1 * (6/2.7706) ≈ 2.00
        result = s.mineral.to_cation(n_oxygens=6, total_oxygens=2.7706)
        assert result.iloc[0] == pytest.approx(2.00, abs=0.02)
