"""Tests for PetroAccessor (df.petro), MolesAccessor (df.moles),
ApfuAccessor (df.apfu), and OxidesAccessor (df.oxides) roundtrips."""

from __future__ import annotations

import pandas as pd
import pytest

import petropandas  # noqa: F401 — triggers accessor registration


# ---------------------------------------------------------------------------
# PetroAccessor — init + general tools
# ---------------------------------------------------------------------------


class TestAccessorInit:
    def test_strips_whitespace(self) -> None:
        df = pd.DataFrame({" SiO2 ": [55.0], " MgO ": [20.0]})
        assert "SiO2" in df.mineral._obj.columns
        assert "MgO" in df.mineral._obj.columns

    def test_renames_aliases(self) -> None:
        df = pd.DataFrame({"FeO*": [10.0], "SiO2": [50.0]})
        assert "FeO" in df.mineral._obj.columns
        assert "FeO*" not in df.mineral._obj.columns

    def test_alias_and_whitespace(self) -> None:
        df = pd.DataFrame({" FeO* ": [10.0], " SiO2 ": [50.0]})
        assert "FeO" in df.mineral._obj.columns
        assert "SiO2" in df.mineral._obj.columns

    def test_nan_replaced_with_zero(self) -> None:
        df = pd.DataFrame({"SiO2": [50.0, float("nan")], "FeO": [10.0, 20.0]})
        assert df.mineral._obj["SiO2"].iloc[1] == 0.0

    def test_negatives_replaced_with_zero(self) -> None:
        df = pd.DataFrame({"SiO2": [-5.0], "FeO": [20.0]})
        assert df.mineral._obj["SiO2"].iloc[0] == 0.0

    def test_non_oxide_columns_untouched(self) -> None:
        df = pd.DataFrame({"SiO2": [50.0], "label": ["foo"]})
        assert df.mineral._obj["label"].iloc[0] == "foo"

    def test_skips_cleanup_when_petro_units_set(self) -> None:
        df = pd.DataFrame({"SiO2": [50.0], "FeO": [10.0]})
        df.attrs["petro_units"] = "wt%"
        accessor = df.mineral
        assert (accessor._obj["SiO2"] == pd.Series([50.0])).all()


# ---------------------------------------------------------------------------
# MolesAccessor (df.moles)
# ---------------------------------------------------------------------------


class TestMolesAccessor:
    def test_from_wt(self, diopside: pd.DataFrame) -> None:
        result = diopside.moles()
        assert result["SiO2"].iloc[0] == pytest.approx(0.9235, abs=0.001)

    def test_sets_attrs(self, diopside: pd.DataFrame) -> None:
        result = diopside.moles()
        assert result.attrs.get("petro_units") == "moles"

    def test_from_moles_idempotent(self, diopside: pd.DataFrame) -> None:
        m1 = diopside.moles()
        m2 = m1.moles()
        pd.testing.assert_frame_equal(m1, m2)

    def test_from_moles_input(self, diopside: pd.DataFrame) -> None:
        m = diopside.moles()
        m2 = m.moles()
        pd.testing.assert_frame_equal(m, m2)

    def test_from_apfu(self, diopside: pd.DataFrame) -> None:
        result = diopside.apfu(n_oxygens=6).moles()
        assert result.attrs.get("petro_units") == "moles"
        assert "SiO2" in result.columns


# ---------------------------------------------------------------------------
# OxidesAccessor roundtrips (df.oxides)
# ---------------------------------------------------------------------------


class TestOxidesRoundtrip:
    def test_from_wt(self, diopside: pd.DataFrame) -> None:
        result = diopside.oxides()
        assert result.attrs.get("petro_units") == "wt%"
        assert "SiO2" in result.columns

    def test_from_wt_idempotent(self, diopside: pd.DataFrame) -> None:
        ox1 = diopside.oxides()
        ox2 = ox1.oxides()
        pd.testing.assert_frame_equal(ox1, ox2)

    def test_from_moles(self, diopside: pd.DataFrame) -> None:
        back = diopside.moles().oxides()
        pd.testing.assert_frame_equal(back, diopside[["SiO2", "MgO", "CaO"]])

    def test_from_moles_roundtrip(self, diopside: pd.DataFrame) -> None:
        back = diopside.oxides().moles().oxides()
        pd.testing.assert_frame_equal(back, diopside[["SiO2", "MgO", "CaO"]])

    def test_roundtrip_fe(self, fe_pyroxene: pd.DataFrame) -> None:
        back = fe_pyroxene.oxides().moles().oxides()
        pd.testing.assert_frame_equal(back, fe_pyroxene)

    def test_sets_attrs(self, diopside: pd.DataFrame) -> None:
        result = diopside.oxides()
        assert result.attrs.get("petro_units") == "wt%"

    def test_no_oxide_columns(self) -> None:
        df = pd.DataFrame({"label": ["a"], "value": [1]})
        result = df.oxides()
        assert result.empty

    def test_from_apfu_roundtrip_ratios(self, diopside: pd.DataFrame) -> None:
        ox = diopside[["SiO2", "MgO", "CaO"]]
        back = ox.apfu(n_oxygens=6).oxides()
        for c in ox.columns:
            r_orig = ox[c].iloc[0] / ox.sum(axis=1).iloc[0]
            r_back = back[c].iloc[0] / back.sum(axis=1).iloc[0]
            assert r_back == pytest.approx(r_orig, rel=1e-4)

    def test_from_apfu_exact_roundtrip(self, diopside: pd.DataFrame) -> None:
        ox = diopside[["SiO2", "MgO", "CaO"]]
        back = ox.apfu(n_oxygens=6).oxides()
        for c in ox.columns:
            assert back[c].iloc[0] == pytest.approx(ox[c].iloc[0], rel=1e-4)

    def test_from_apfu_multirow(self) -> None:
        df = pd.DataFrame(
            {
                "SiO2": [44.116, 53.129],
                "Al2O3": [10.095, 3.224],
                "FeO": [15.474, 8.835],
                "MgO": [11.329, 17.624],
                "CaO": [10.765, 11.361],
                "Na2O": [2.582, 0.992],
            }
        )
        back = df.apfu(n_oxygens=12).oxides()
        for c in df.columns:
            pd.testing.assert_series_equal(back[c], df[c], check_names=False, atol=1e-2)

    def test_from_apfu_attrs(self, diopside: pd.DataFrame) -> None:
        apfu = diopside.apfu(n_oxygens=6)
        assert apfu.attrs.get("petro_total") is not None
        result = apfu.oxides()
        assert result.attrs.get("petro_units") == "wt%"


class TestOxidesNormalized:
    def test_sums_to_100(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.normalized()
        assert result.sum(axis=1).iloc[0] == pytest.approx(100.0)

    def test_preserves_units(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.normalized()
        assert result.attrs.get("petro_units") == "wt%"


# ---------------------------------------------------------------------------
# ApfuAccessor (df.apfu)
# ---------------------------------------------------------------------------


class TestApfuAccessor:
    def test_diopside_oxygen(self, diopside: pd.DataFrame) -> None:
        result = diopside.apfu(n_oxygens=6)
        assert "Si{4+}" in result.columns
        assert "Mg{2+}" in result.columns
        assert "Ca{2+}" in result.columns
        assert result["Si{4+}"].iloc[0] == pytest.approx(2.00, abs=0.01)
        assert result["Mg{2+}"].iloc[0] == pytest.approx(1.00, abs=0.01)
        assert result["Ca{2+}"].iloc[0] == pytest.approx(1.00, abs=0.01)

    def test_diopside_cation(self, diopside: pd.DataFrame) -> None:
        result = diopside.apfu(n_cations=4)
        assert result["Si{4+}"].iloc[0] == pytest.approx(2.00, abs=0.01)

    def test_from_wt(self, fe_pyroxene: pd.DataFrame) -> None:
        from_wt = fe_pyroxene.apfu(n_oxygens=6)
        assert from_wt.attrs.get("petro_units") == "apfu"

    def test_from_moles(self, fe_pyroxene: pd.DataFrame) -> None:
        from_wt = fe_pyroxene.apfu(n_oxygens=6)
        from_moles = fe_pyroxene.moles().apfu(n_oxygens=6)
        pd.testing.assert_frame_equal(from_wt, from_moles)

    def test_sets_attrs(self, diopside: pd.DataFrame) -> None:
        result = diopside.apfu(n_oxygens=6)
        assert result.attrs.get("petro_units") == "apfu"
        assert result.attrs.get("petro_n_oxygens") == 6
        assert result.attrs.get("petro_n_cations") is None

    def test_cation_attrs(self, diopside: pd.DataFrame) -> None:
        result = diopside.apfu(n_cations=4)
        assert result.attrs.get("petro_n_cations") == 4
        assert result.attrs.get("petro_n_oxygens") is None

    def test_idempotent(self, diopside: pd.DataFrame) -> None:
        a1 = diopside.apfu(n_oxygens=6)
        a2 = a1.apfu(n_oxygens=6)
        pd.testing.assert_frame_equal(a1, a2)


# ---------------------------------------------------------------------------
# OxidesAccessor.split_valence (df.oxides.split_valence)
# ---------------------------------------------------------------------------


class TestOxidesSplitValence:
    def test_droop_fe(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.split_valence(
            "Fe", method="droop", n_oxygens=6, ideal_cations=4
        )
        assert "FeO" in result.columns
        assert "Fe2O3" in result.columns
        assert (result["FeO"] >= 0).all()
        assert (result["Fe2O3"] >= 0).all()

    def test_sets_attrs(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.split_valence(
            "Fe", method="droop", n_oxygens=6, ideal_cations=4
        )
        assert result.attrs.get("petro_units") == "wt%"

    def test_preserves_other_columns(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.split_valence(
            "Fe", method="droop", n_oxygens=6, ideal_cations=4
        )
        assert "SiO2" in result.columns
        assert "MgO" in result.columns


# ---------------------------------------------------------------------------
# OxidesAccessor.oxidize (df.oxides.oxidize)
# ---------------------------------------------------------------------------


class TestOxidesOxidize:
    def test_basic_split(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.oxidize(o_excess=0.05)
        assert "FeO" in result.columns
        assert "Fe2O3" in result.columns
        assert (result["FeO"] >= 0).all()
        assert (result["Fe2O3"] >= 0).all()
        assert result["Fe2O3"].iloc[0] > 0

    def test_sets_attrs(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.oxidize(o_excess=0.05)
        assert result.attrs.get("petro_units") == "wt%"

    def test_preserves_other_columns(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.oxidize(o_excess=0.05)
        assert "SiO2" in result.columns
        assert "MgO" in result.columns

    def test_idempotent(self, fe_pyroxene: pd.DataFrame) -> None:
        first = fe_pyroxene.oxides.oxidize(o_excess=0.05)
        second = first.oxides.oxidize(o_excess=0.05)
        pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# OxidesAccessor.reduce (df.oxides.reduce)
# ---------------------------------------------------------------------------


class TestOxidesReduce:
    def test_basic(self, spinel: pd.DataFrame) -> None:
        result = spinel.oxides.reduce()
        assert "FeO" in result.columns
        assert "Fe2O3" not in result.columns
        assert (result["FeO"] > 0).all()

    def test_sets_attrs(self, spinel: pd.DataFrame) -> None:
        result = spinel.oxides.reduce()
        assert result.attrs.get("petro_units") == "wt%"

    def test_preserves_other_columns(self, spinel: pd.DataFrame) -> None:
        result = spinel.oxides.reduce()
        assert "SiO2" in result.columns
        assert "Al2O3" in result.columns
        assert "MgO" in result.columns

    def test_no_fe2o3_silent(self, fe_pyroxene: pd.DataFrame) -> None:
        result = fe_pyroxene.oxides.reduce()
        pd.testing.assert_frame_equal(result, fe_pyroxene.oxides())
