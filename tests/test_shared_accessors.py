"""Tests for `mean`/`reframe`/`normalize`/`select` shared via `_BaseAccessor`."""

from __future__ import annotations

import pandas as pd
import pytest

import petropandas  # noqa: F401 — triggers accessor registration
from petropandas._accessors import OxidesAccessor


class TestOxidesMethodsNotRedefined:
    def test_mean_is_inherited(self) -> None:
        assert "mean" not in OxidesAccessor.__dict__

    def test_select_is_inherited(self) -> None:
        assert "select" not in OxidesAccessor.__dict__


class TestMineralAccessorHasNoBaseAccessorMethods:
    def test_no_mean(self, garnet_multi: pd.DataFrame) -> None:
        assert not hasattr(garnet_multi.mineral, "mean")

    def test_no_reframe(self, garnet_multi: pd.DataFrame) -> None:
        assert not hasattr(garnet_multi.mineral, "reframe")

    def test_no_normalize(self, garnet_multi: pd.DataFrame) -> None:
        assert not hasattr(garnet_multi.mineral, "normalize")

    def test_no_select(self, garnet_multi: pd.DataFrame) -> None:
        assert not hasattr(garnet_multi.mineral, "select")


class TestMeanAcrossUnits:
    def test_moles_mean_units(self, garnet_multi: pd.DataFrame) -> None:
        moles = garnet_multi.moles()
        result = moles.moles.mean()
        assert result.attrs.get("petro_units") == "moles"

    def test_moles_mean_matches_manual(self, garnet_multi: pd.DataFrame) -> None:
        moles = garnet_multi.moles()
        result = moles.moles.mean()
        for col in moles.columns:
            assert result[col].iloc[0] == pytest.approx(moles[col].mean())

    def test_cations_mean_units(self, garnet_multi: pd.DataFrame) -> None:
        apfu = garnet_multi.cations(n_oxygens=12)
        result = apfu.cations.mean()
        assert result.attrs.get("petro_units") == "apfu"

    def test_groupby_missing_column_raises(self, garnet_multi: pd.DataFrame) -> None:
        moles = garnet_multi.moles()
        with pytest.raises(ValueError, match="not found"):
            moles.moles.mean(groupby="missing")


class TestReframeAcrossUnits:
    def test_moles_reframe_units(self, garnet_multi: pd.DataFrame) -> None:
        moles = garnet_multi.moles()
        result = moles.moles.reframe(["SiO2", "Al2O3", "TiO2"])
        assert result.attrs.get("petro_units") == "moles"
        assert list(result.columns) == ["SiO2", "Al2O3", "TiO2"]
        assert (result["TiO2"] == 0.0).all()

    def test_oxides_reframe_units(self, garnet_multi: pd.DataFrame) -> None:
        oxides = garnet_multi.oxides()
        result = oxides.oxides.reframe(["SiO2", "F"])
        assert result.attrs.get("petro_units") == "wt%"


class TestNormalizeAcrossUnits:
    def test_oxides_normalize_units(self, garnet_multi: pd.DataFrame) -> None:
        result = garnet_multi.oxides.normalize()
        assert result.attrs.get("petro_units") == "wt%"
        assert result.sum(axis=1).iloc[0] == pytest.approx(100.0)

    def test_cations_normalize_units(self, garnet_multi: pd.DataFrame) -> None:
        apfu = garnet_multi.cations(n_oxygens=12)
        result = apfu.cations.normalize()
        assert result.attrs.get("petro_units") == "apfu"

    def test_to_kwarg(self, garnet_multi: pd.DataFrame) -> None:
        result = garnet_multi.oxides.normalize(to=1.0)
        assert result.sum(axis=1).iloc[0] == pytest.approx(1.0)


class TestSelectAcrossUnits:
    def test_moles_select(self, garnet_multi: pd.DataFrame) -> None:
        moles = garnet_multi.moles()
        result = moles.moles.select([0, 2])
        assert list(result.index) == [0, 2]
        assert result.attrs.get("petro_units") == "moles"

    def test_bulk_select(self, garnet_multi: pd.DataFrame) -> None:
        result = garnet_multi.bulk.select([0, 2])
        assert list(result.index) == [0, 2]
        assert result.attrs.get("petro_units") == "wt%"
