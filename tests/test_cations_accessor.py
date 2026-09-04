"""Tests for `CationsAccessor.total_charge`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import petropandas  # noqa: F401 — triggers accessor registration
from petropandas import _calc


class TestTotalCharge:
    def test_matches_calc(self, garnet_multi: pd.DataFrame) -> None:
        apfu = garnet_multi.cations(n_oxygens=12)
        assert apfu.cations.total_charge().equals(_calc.total_charge(apfu))

    def test_returns_series(self, garnet_multi: pd.DataFrame) -> None:
        apfu = garnet_multi.cations(n_oxygens=12)
        assert isinstance(apfu.cations.total_charge(), pd.Series)

    def test_name(self, garnet_multi: pd.DataFrame) -> None:
        apfu = garnet_multi.cations(n_oxygens=12)
        assert apfu.cations.total_charge().name == "total_charge"

    def test_manual_value(self) -> None:
        apfu = pd.DataFrame({"Si{4+}": [2.0], "Fe{2+}": [1.0], "Al{3+}": [2.0]})
        apfu.attrs["petro_units"] = "apfu"
        result = apfu.cations.total_charge()
        assert result.iloc[0] == pytest.approx(2 * 4 + 1 * 2 + 2 * 3)

    def test_consistent_with_score_charge_balance(
        self, garnet_multi: pd.DataFrame
    ) -> None:
        apfu = garnet_multi.cations(n_oxygens=12)
        expected = np.exp(-(apfu.cations.total_charge() - 24.0).abs() / 0.5)
        pd.testing.assert_series_equal(
            _calc.score_charge_balance(apfu, n_oxygens=12),
            expected,
            check_names=False,
        )
