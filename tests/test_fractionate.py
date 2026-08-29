"""Tests for radial-profile integration and BulkAccessor.fractionate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from petropandas import Grt, Mineral, _calc
from petropandas._core import MW, _cations_per
from petropandas.data import bulk as bulk_data
from petropandas.data import grt_profile

# ---------------------------------------------------------------------------
# integrate_radial_profile
# ---------------------------------------------------------------------------


def _linear_profile(n_points: int, a: float, b: float) -> pd.DataFrame:
    r = np.arange(n_points, dtype=float)
    return pd.DataFrame({"SiO2": a + b * r, "MgO": 100.0 - (a + b * r)}, index=r)


class TestIntegrateRadialProfile:
    def test_core_to_rim_matches_closed_form(self) -> None:
        a, b, n = 10.0, 2.0, 21
        profile = _linear_profile(n, a, b)
        r_max = n - 1
        result = _calc.integrate_radial_profile(
            profile, order="core-to-rim", n_grid=3000
        )
        expected_sio2 = a + 0.75 * b * r_max
        assert result["SiO2"] == pytest.approx(expected_sio2, rel=1e-3)

    def test_rim_to_core_matches_core_to_rim(self) -> None:
        a, b, n = 10.0, 2.0, 21
        profile = _linear_profile(n, a, b)
        core_to_rim = _calc.integrate_radial_profile(
            profile, order="core-to-rim", n_grid=2000
        )
        reversed_profile = profile.iloc[::-1]
        rim_to_core = _calc.integrate_radial_profile(
            reversed_profile, order="rim-to-core", n_grid=2000
        )
        pd.testing.assert_series_equal(core_to_rim, rim_to_core, rtol=1e-6)

    def test_rim_core_rim_matches_single_direction(self) -> None:
        a, b, half_n = 10.0, 2.0, 15
        half = _linear_profile(half_n, a, b)
        single_direction = _calc.integrate_radial_profile(
            half, order="core-to-rim", n_grid=2000
        )

        # Mirror the half profile around its first row (the core, r=0) to
        # build a full rim -> core -> rim traverse with a single shared
        # core point at the center.
        mirrored_rim = half.iloc[1:].iloc[::-1]
        full = pd.concat([mirrored_rim, half], ignore_index=True)
        full.index = np.arange(len(full), dtype=float)

        result = _calc.integrate_radial_profile(full, order="rim-core-rim", n_grid=2000)
        pd.testing.assert_series_equal(single_direction, result, rtol=1e-6)

    def test_invalid_order_raises(self) -> None:
        profile = _linear_profile(5, 10.0, 1.0)
        with pytest.raises(ValueError, match="order"):
            _calc.integrate_radial_profile(profile, order="sideways")


# ---------------------------------------------------------------------------
# _calc.fractionate — molar mass balance
# ---------------------------------------------------------------------------


def _manual_fractionate(
    bulk_row: dict[str, float],
    mineral_row: dict[str, float],
    f_oxide: float,
) -> dict[str, float]:
    """Independent scalar reimplementation of the cation-mole mass balance."""
    cols = list(bulk_row)
    mw = {c: MW(c) for c in cols}
    cpo = {c: _cations_per(c) for c in cols}
    n_bulk = {c: bulk_row[c] / mw[c] * cpo[c] for c in cols}
    n_gnt = {c: mineral_row[c] / mw[c] * cpo[c] for c in cols}
    n_bulk_total = sum(n_bulk.values())
    n_gnt_total = sum(n_gnt.values())
    x_bulk = {c: n_bulk[c] / n_bulk_total for c in cols}
    x_gnt = {c: n_gnt[c] / n_gnt_total for c in cols}
    x_eff = {
        c: max(0.0, (x_bulk[c] - f_oxide * x_gnt[c]) / (1.0 - f_oxide)) for c in cols
    }
    m_eff = {c: x_eff[c] / cpo[c] * mw[c] for c in cols}
    total_m = sum(m_eff.values())
    original_total = sum(bulk_row.values())
    return {c: m_eff[c] / total_m * original_total for c in cols}


class TestCalcFractionate:
    def test_mineral_none_matches_manual_calc(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0], "MgO": [40.0]})
        mineral_wt = pd.Series({"SiO2": 40.0, "MgO": 60.0})
        fraction = pd.Series([0.2], index=bulk.index)

        result = _calc.fractionate(bulk, mineral_wt, fraction, ideal_cations=None)

        expected = _manual_fractionate(
            {"SiO2": 60.0, "MgO": 40.0}, {"SiO2": 40.0, "MgO": 60.0}, 0.2
        )
        for col, val in expected.items():
            assert result[col].iloc[0] == pytest.approx(val)

    def test_ideal_cations_scaling_matches_manual_calc(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0], "Al2O3": [15.0], "MgO": [25.0]})
        mineral_wt = pd.Series({"SiO2": 38.0, "Al2O3": 21.0, "MgO": 5.0})
        f_fu = 0.03
        ideal_cations = 8.0

        result = _calc.fractionate(
            bulk,
            mineral_wt,
            pd.Series([f_fu], index=bulk.index),
            ideal_cations=ideal_cations,
        )

        # f_oxide = ideal_cations * f_fu / N_bulk (cation-mole basis)
        cols = ["SiO2", "Al2O3", "MgO"]
        cpo = {c: _cations_per(c) for c in cols}
        n_bulk_total = sum(bulk[c].iloc[0] / MW(c) * cpo[c] for c in cols)
        f_oxide = ideal_cations * f_fu / n_bulk_total

        expected = _manual_fractionate(
            {c: bulk[c].iloc[0] for c in cols},
            {c: mineral_wt[c] for c in cols},
            f_oxide,
        )
        for col, val in expected.items():
            assert result[col].iloc[0] == pytest.approx(val)

    def test_renormalizes_to_original_total(self) -> None:
        bulk = pd.DataFrame({"SiO2": [70.0], "Al2O3": [20.0], "FeO": [5.0]})
        mineral_wt = pd.Series({"SiO2": 38.0, "Al2O3": 21.0, "FeO": 30.0})
        result = _calc.fractionate(
            bulk, mineral_wt, pd.Series([0.05], index=bulk.index)
        )
        assert result.sum(axis=1).iloc[0] == pytest.approx(95.0)

    def test_large_fraction_clips_at_zero(self) -> None:
        bulk = pd.DataFrame({"SiO2": [90.0], "MgO": [10.0]})
        mineral_wt = pd.Series({"SiO2": 5.0, "MgO": 95.0})
        result = _calc.fractionate(
            bulk, mineral_wt, pd.Series([0.99], index=bulk.index)
        )
        assert (result >= 0.0).all().all()


# ---------------------------------------------------------------------------
# BulkAccessor.fractionate
# ---------------------------------------------------------------------------


class TestBulkAccessorFractionate:
    def test_matches_calc_function(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0], "MgO": [40.0]})
        profile = _linear_profile(11, 30.0, 1.0)

        accessor_result = bulk.bulk.fractionate(profile, fraction=[0.1])

        mineral_wt = _calc.integrate_radial_profile(profile)
        expected = _calc.fractionate(
            bulk, mineral_wt, pd.Series([0.1], index=bulk.index), ideal_cations=None
        )
        pd.testing.assert_frame_equal(accessor_result, expected, check_like=True)

    def test_fraction_column_name(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0], "MgO": [40.0], "frac": [0.1]})
        profile = _linear_profile(11, 30.0, 1.0)
        by_column = bulk.bulk.fractionate(profile, fraction="frac")
        by_list = bulk.bulk.fractionate(profile, fraction=[0.1])
        pd.testing.assert_frame_equal(by_column, by_list, check_like=True)

    def test_fraction_float_broadcasts_to_all_rows(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0, 55.0], "MgO": [40.0, 45.0]})
        profile = _linear_profile(11, 30.0, 1.0)
        by_float = bulk.bulk.fractionate(profile, fraction=0.1)
        by_list = bulk.bulk.fractionate(profile, fraction=[0.1, 0.1])
        pd.testing.assert_frame_equal(by_float, by_list, check_like=True)

    def test_missing_fraction_column_raises(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0], "MgO": [40.0]})
        profile = _linear_profile(11, 30.0, 1.0)
        with pytest.raises(ValueError, match="not found"):
            bulk.bulk.fractionate(profile, fraction="missing")

    def test_mismatched_length_fraction_raises(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0, 55.0], "MgO": [40.0, 45.0]})
        profile = _linear_profile(11, 30.0, 1.0)
        with pytest.raises(ValueError, match="length"):
            bulk.bulk.fractionate(profile, fraction=[0.1])

    def test_mineral_without_ideal_cations_raises(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0], "MgO": [40.0]})
        profile = _linear_profile(11, 30.0, 1.0)
        bare_mineral = Mineral()
        with pytest.raises(ValueError, match="ideal_cations"):
            bulk.bulk.fractionate(profile, fraction=[0.03], mineral=bare_mineral)

    def test_attrs_petro_units(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0], "MgO": [40.0]})
        profile = _linear_profile(11, 30.0, 1.0)
        result = bulk.bulk.fractionate(profile, fraction=[0.1])
        assert result.attrs.get("petro_units") == "wt%"

    def test_with_mineral_kwarg(self) -> None:
        bulk = pd.DataFrame({"SiO2": [60.0], "Al2O3": [15.0], "FeO": [25.0]})
        profile = pd.DataFrame(
            {
                "SiO2": [37.0, 37.5, 38.0],
                "Al2O3": [21.0, 21.2, 21.4],
                "FeO": [30.0, 29.0, 28.0],
            }
        )
        result = bulk.bulk.fractionate(profile, fraction=[0.03], mineral=Grt)
        assert result.sum(axis=1).iloc[0] == pytest.approx(bulk.sum(axis=1).iloc[0])
        assert (result >= 0.0).all().all()

    def test_bundled_example_data_smoke(self) -> None:
        one_bulk = bulk_data.iloc[[0]]
        result = one_bulk.bulk.fractionate(
            grt_profile, fraction=[0.03], mineral=Grt, order="rim-core-rim"
        )
        assert not result.isna().any().any()
        assert (result >= 0.0).all().all()
        original_total = one_bulk.bulk()
        original_total = original_total[
            [c for c in original_total.columns if c in result.columns]
        ].sum(axis=1)
        assert result.sum(axis=1).iloc[0] == pytest.approx(original_total.iloc[0])
