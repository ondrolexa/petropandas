import numpy as np
import pytest

from petropandas import pd
from petropandas.minerals import (
    DiMica,
    Feldspar,
    Garnet,
    Garnet_Fe2,
    Garnet_TC,
    Garnet_TCnoMn,
    Mineral,
    MineralNotCalculated,
    Pyroxene,
    Pyroxene_Fe2,
    Site,
    StrucForm,
    formula2wt,
)

# ==========================================================
# Site
# ==========================================================


class TestSite:
    def test_create(self):
        s = Site("X", 3, ["Mg{2+}", "Fe{2+}", "Ca{2+}"])
        assert s.name == "X"
        assert s.ncat == 3
        assert s.free == 3

    def test_add_atom_under_capacity(self):
        s = Site("X", 3, ["Mg{2+}", "Fe{2+}"])
        s.add("Mg{2+}", 1.5)
        assert s.get("Mg{2+}") == 1.5
        assert s.free == 1.5

    def test_add_atom_clamps_to_free(self):
        s = Site("X", 3, ["Mg{2+}", "Fe{2+}"])
        s.add("Mg{2+}", 5.0)
        assert s.get("Mg{2+}") == 3.0
        assert s.free == 0.0

    def test_add_atom_force(self):
        s = Site("X", 3, ["Mg{2+}", "Fe{2+}"])
        s.add("Mg{2+}", 5.0, force=True)
        assert s.get("Mg{2+}") == 5.0
        assert s.free == -2.0

    def test_add_two_atoms(self):
        s = Site("X", 3, ["Mg{2+}", "Fe{2+}"])
        s.add("Mg{2+}", 1.0)
        s.add("Fe{2+}", 1.0)
        assert s.get("Mg{2+}") == 1.0
        assert s.get("Fe{2+}") == 1.0
        assert s.free == 1.0

    def test_get_fraction(self):
        s = Site("X", 4, ["Mg{2+}", "Fe{2+}"])
        s.add("Mg{2+}", 2.0)
        assert s.get("Mg{2+}", fraction=True) == 0.5

    def test_get_missing_atom(self):
        s = Site("X", 3, ["Mg{2+}", "Fe{2+}"])
        assert s.get("Ca{2+}") == 0

    def test_add_same_atom_twice(self):
        s = Site("X", 3, ["Mg{2+}", "Fe{2+}"])
        s.add("Mg{2+}", 1.0)
        s.add("Mg{2+}", 1.5)
        assert s.get("Mg{2+}") == 2.5
        assert s.free == 0.5

    def test_add_no_room(self):
        s = Site("X", 1, ["Mg{2+}", "Fe{2+}"])
        s.add("Mg{2+}", 1.0)
        s.add("Fe{2+}", 1.0)
        assert s.get("Fe{2+}") == 0.0

    def test_repr(self):
        s = Site("X", 3, ["Mg{2+}"])
        s.add("Mg{2+}", 2.0)
        r = repr(s)
        assert "X" in r
        assert "3" in r
        assert "2" in r


# ==========================================================
# StrucForm
# ==========================================================


class TestStrucForm:
    def test_create(self):
        m = Garnet_Fe2()
        sf = StrucForm(m)
        assert len(sf.sites) == 3
        assert sf.reminder.empty

    def test_get_total_atom(self):
        m = Garnet_Fe2()
        sf = StrucForm(m)
        sf.sites[0].add("Si{4+}", 3.0)
        assert sf.get("Si{4+}") == 3.0

    def test_site_by_name(self):
        m = Garnet_Fe2()
        sf = StrucForm(m)
        assert sf.site("Z") is sf.sites[0]
        assert sf.site("X") is sf.sites[2]

    def test_site_missing(self):
        m = Garnet_Fe2()
        sf = StrucForm(m)
        assert sf.site("W") is None

    def test_calculate(self):
        m = Garnet_Fe2()
        cations = pd.Series(
            {"Si{4+}": 3.0, "Al{3+}": 2.0, "Mg{2+}": 1.0, "Fe{2+}": 1.0, "Ca{2+}": 1.0}
        )
        sf = m.calculate(cations)
        assert not sf.reminder.empty
        apfu = sf.apfu
        assert apfu["Si{4+}"] == pytest.approx(3.0)
        assert apfu["Al{3+}"] == pytest.approx(2.0)

    def test_apfu_before_calculate_raises(self):
        m = Garnet_Fe2()
        sf = StrucForm(m)
        with pytest.raises(MineralNotCalculated):
            _ = sf.apfu

    def test_check_stechiometry_before_calculate_raises(self):
        m = Garnet_Fe2()
        sf = StrucForm(m)
        with pytest.raises(MineralNotCalculated):
            _ = sf.check_stechiometry

    def test_check_stechiometry_calculated(self):
        m = Garnet_Fe2()
        cations = pd.Series(
            {"Si{4+}": 3.0, "Al{3+}": 2.0, "Mg{2+}": 1.0, "Fe{2+}": 1.0, "Ca{2+}": 1.0}
        )
        sf = m.calculate(cations)
        misfit = sf.check_stechiometry
        assert isinstance(misfit, float | np.floating)

    def test_reminder_nonzero_without_force(self):
        m = Garnet_Fe2()
        cations = pd.Series(
            {"Si{4+}": 3.0, "Al{3+}": 2.0, "Mg{2+}": 5.0, "Fe{2+}": 0.0, "Ca{2+}": 0.0}
        )
        sf = m.calculate(cations)
        assert sf.reminder["Mg{2+}"] > 0

    def test_repr_before(self):
        m = Garnet_Fe2()
        sf = StrucForm(m)
        assert "Not yet" in repr(sf)

    def test_repr_after_calc(self):
        m = Garnet_Fe2()
        cations = pd.Series(
            {"Si{4+}": 3.0, "Al{3+}": 2.0, "Mg{2+}": 1.0, "Fe{2+}": 1.0, "Ca{2+}": 1.0}
        )
        sf = m.calculate(cations)
        r = repr(sf)
        assert "Si" in r


# ==========================================================
# Mineral base
# ==========================================================


class TestMineral:
    def test_ncat_computed(self):
        m = Garnet_Fe2()
        assert m.ncat == 8

    def test_has_endmembers_defined_subclass(self):
        m = Garnet_Fe2()
        assert m.has_endmembers

    def test_has_structure_defined(self):
        m = Garnet_Fe2()
        assert m.has_structure

    def test_no_structure(self):
        m = Mineral()
        assert not m.has_structure

    def test_endmembers_not_implemented(self):
        m = Mineral()
        with pytest.raises(NotImplementedError):
            m.endmembers(pd.Series({"Si{4+}": 1.0}))

    def test_apfu(self):
        m = Feldspar()
        cations = pd.Series(
            {"Si{4+}": 3.0, "Al{3+}": 1.0, "K{+}": 0.5, "Na{+}": 0.3, "Ca{2+}": 0.2}
        )
        apfu = m.apfu(cations)
        assert isinstance(apfu, pd.Series)

    def test_check_stechiometry(self):
        m = Feldspar()
        cations = pd.Series(
            {"Si{4+}": 3.0, "Al{3+}": 1.0, "K{+}": 0.5, "Na{+}": 0.3, "Ca{2+}": 0.2}
        )
        misfit = m.check_stechiometry(cations)
        assert isinstance(misfit, float | np.floating)

    def test_repr(self):
        m = Garnet_Fe2()
        assert repr(m) == "Garnet_Fe2"


# ==========================================================
# formula2wt
# ==========================================================


class TestFormula2wt:
    def test_simple_silicate(self):
        result = formula2wt("Mg2SiO4")
        assert "MgO" in result.columns
        assert "SiO2" in result.columns
        assert result["MgO"].iloc[0] > 0
        assert result["SiO2"].iloc[0] > 0

    def test_oxide_mapping(self):
        result = formula2wt("CaAl2Si2O8")
        assert "CaO" in result.columns
        assert "Al2O3" in result.columns
        assert "SiO2" in result.columns

    def test_unknown_element_skipped(self):
        result = formula2wt("FeS2")
        assert "FeO" in result.columns
        assert "S" not in result.columns


# ==========================================================
# Garnet_Fe2
# ==========================================================


class TestGarnetFe2:
    def test_noxy_ncat(self):
        m = Garnet_Fe2()
        assert m.noxy == 12
        assert m.ncat == 8
        assert m.needsFe == "Fe2"

    def test_endmembers_sum_to_one(self):
        m = Garnet_Fe2()
        cations = pd.Series(
            {
                "Si{4+}": 3.0,
                "Al{3+}": 2.0,
                "Mg{2+}": 1.0,
                "Fe{2+}": 1.5,
                "Mn{2+}": 0.3,
                "Ca{2+}": 0.2,
            }
        )
        em = m.endmembers(cations)
        assert list(em.index) == ["Alm", "Prp", "Sps", "Grs"]
        assert em.sum() == pytest.approx(1.0)

    def test_endmembers_pure_almandine(self):
        m = Garnet_Fe2()
        cations = pd.Series(
            {
                "Si{4+}": 3.0,
                "Al{3+}": 2.0,
                "Fe{2+}": 3.0,
                "Mg{2+}": 0.0,
                "Mn{2+}": 0.0,
                "Ca{2+}": 0.0,
            }
        )
        em = m.endmembers(cations)
        assert em["Alm"] == pytest.approx(1.0)
        assert em["Prp"] == pytest.approx(0.0)
        assert em["Grs"] == pytest.approx(0.0)

    def test_endmembers_grossular(self):
        m = Garnet_Fe2()
        cations = pd.Series(
            {
                "Si{4+}": 3.0,
                "Al{3+}": 2.0,
                "Ca{2+}": 3.0,
                "Mg{2+}": 0.0,
                "Mn{2+}": 0.0,
                "Fe{2+}": 0.0,
            }
        )
        em = m.endmembers(cations)
        assert em["Grs"] == pytest.approx(1.0)


# ==========================================================
# Garnet (Fe3)
# ==========================================================


class TestGarnet:
    def test_noxy_ncat(self):
        m = Garnet()
        assert m.noxy == 12
        assert m.needsFe == "Fe3"

    def test_endmembers_sum_to_one(self):
        m = Garnet()
        cations = pd.Series(
            {
                "Si{4+}": 3.0,
                "Al{3+}": 1.5,
                "Fe{3+}": 0.5,
                "Ti{4+}": 0.1,
                "Mg{2+}": 1.0,
                "Fe{2+}": 1.0,
                "Mn{2+}": 0.3,
                "Ca{2+}": 0.7,
            }
        )
        em = m.endmembers(cations)
        assert em.sum() == pytest.approx(1.0)

    def test_endmembers_all_ca(self):
        m = Garnet()
        cations = pd.Series(
            {
                "Si{4+}": 3.0,
                "Al{3+}": 2.0,
                "Fe{3+}": 0.0,
                "Ti{4+}": 0.0,
                "Cr{3+}": 0.0,
                "Y{3+}": 0.0,
                "Mg{2+}": 0.0,
                "Fe{2+}": 0.0,
                "Mn{2+}": 0.0,
                "Ca{2+}": 3.0,
                "Na{+}": 0.0,
            }
        )
        em = m.endmembers(cations)
        assert em["Grs"] == pytest.approx(1.0)


# ==========================================================
# Feldspar
# ==========================================================


class TestFeldspar:
    def test_noxy_ncat(self):
        m = Feldspar()
        assert m.noxy == 8
        assert m.ncat == 5
        assert m.needsFe is None

    def test_endmembers_sum_to_one(self):
        m = Feldspar()
        cations = pd.Series(
            {"Si{4+}": 3.0, "Al{3+}": 1.0, "K{+}": 0.5, "Na{+}": 0.3, "Ca{2+}": 0.2}
        )
        em = m.endmembers(cations)
        assert list(em.index) == ["An", "Ab", "Or"]
        assert em.sum() == pytest.approx(1.0)

    def test_endmembers_pure_albite(self):
        m = Feldspar()
        cations = pd.Series(
            {"Si{4+}": 3.0, "Al{3+}": 1.0, "Na{+}": 1.0, "K{+}": 0.0, "Ca{2+}": 0.0}
        )
        em = m.endmembers(cations)
        assert em["Ab"] == pytest.approx(1.0)
        assert em["An"] == pytest.approx(0.0)
        assert em["Or"] == pytest.approx(0.0)

    def test_apfu_known_input(self):
        m = Feldspar()
        cations = pd.Series({"Si{4+}": 3.0, "Al{3+}": 1.0, "Na{+}": 1.0})
        apfu = m.apfu(cations)
        assert apfu["Si{4+}"] == pytest.approx(3.0)
        assert apfu["Al{3+}"] == pytest.approx(1.0)
        assert apfu["Na{+}"] == pytest.approx(1.0)


# ==========================================================
# Pyroxene_Fe2
# ==========================================================


class TestPyroxeneFe2:
    def test_noxy_ncat(self):
        m = Pyroxene_Fe2()
        assert m.noxy == 6
        assert m.ncat == 4
        assert m.needsFe == "Fe2"

    def test_endmembers_sum_to_one(self):
        m = Pyroxene_Fe2()
        cations = pd.Series(
            {"Si{4+}": 2.0, "Mg{2+}": 0.8, "Fe{2+}": 0.6, "Ca{2+}": 0.4, "Mn{2+}": 0.2}
        )
        em = m.endmembers(cations)
        assert list(em.index) == ["En", "Wo", "Fs"]
        assert em.sum() == pytest.approx(1.0)

    def test_endmembers_pure_enstatite(self):
        m = Pyroxene_Fe2()
        cations = pd.Series({"Si{4+}": 2.0, "Mg{2+}": 2.0})
        em = m.endmembers(cations)
        assert em["En"] == pytest.approx(1.0)
        assert em["Wo"] == pytest.approx(0.0)
        assert em["Fs"] == pytest.approx(0.0)


# ==========================================================
# Pyroxene (Fe3, complex)
# ==========================================================


class TestPyroxene:
    def test_noxy_ncat(self):
        m = Pyroxene()
        assert m.noxy == 6
        assert m.needsFe == "Fe3"

    def test_endmembers_runs(self):
        m = Pyroxene()
        cations = pd.Series(
            {
                "Si{4+}": 1.8,
                "Al{3+}": 0.2,
                "Ti{4+}": 0.05,
                "Fe{3+}": 0.1,
                "Cr{3+}": 0.02,
                "Mg{2+}": 0.8,
                "Fe{2+}": 0.6,
                "Mn{2+}": 0.02,
                "Ca{2+}": 0.4,
                "Na{+}": 0.1,
                "K{+}": 0.01,
            }
        )
        em = m.endmembers(cations)
        assert isinstance(em, pd.Series)
        assert len(em) > 0

    def test_endmembers_no_nan(self):
        m = Pyroxene()
        cations = pd.Series(
            {
                "Si{4+}": 2.0,
                "Al{3+}": 0.0,
                "Ti{4+}": 0.0,
                "Fe{3+}": 0.0,
                "Cr{3+}": 0.0,
                "Mg{2+}": 1.0,
                "Fe{2+}": 0.0,
                "Mn{2+}": 0.0,
                "Ca{2+}": 1.0,
                "Na{+}": 0.0,
                "K{+}": 0.0,
            }
        )
        em = m.endmembers(cations)
        assert not em.isna().any()

    def test_endmembers_en_diopside(self):
        m = Pyroxene()
        cations = pd.Series(
            {
                "Si{4+}": 2.0,
                "Al{3+}": 0.0,
                "Ti{4+}": 0.0,
                "Fe{3+}": 0.0,
                "Cr{3+}": 0.0,
                "Mg{2+}": 0.5,
                "Fe{2+}": 0.0,
                "Mn{2+}": 0.0,
                "Ca{2+}": 0.5,
                "Na{+}": 0.0,
                "K{+}": 0.0,
            }
        )
        em = m.endmembers(cations)
        assert em["En"] == pytest.approx(0.25)
        assert em["Wo"] == pytest.approx(0.25)


# ==========================================================
# DiMica
# ==========================================================


class TestDiMica:
    def test_noxy_ncat(self):
        m = DiMica()
        assert m.noxy == 11
        assert m.ncat == 7
        assert m.needsFe == "Fe2"

    def test_endmembers_runs(self):
        m = DiMica()
        cations = pd.Series(
            {
                "Si{4+}": 3.5,
                "Al{3+}": 2.5,
                "Mg{2+}": 0.3,
                "Fe{2+}": 0.3,
                "Mn{2+}": 0.02,
                "Ti{4+}": 0.02,
                "Ca{2+}": 0.02,
                "Na{+}": 0.1,
                "K{+}": 0.8,
            }
        )
        em = m.endmembers(cations)
        assert isinstance(em, pd.Series)
        assert len(em) == 6
        assert not em.isna().any()

    def test_endmembers_no_nan_when_zero_feo_mgo(self):
        m = DiMica()
        cations = pd.Series(
            {
                "Si{4+}": 3.5,
                "Al{3+}": 2.5,
                "Mg{2+}": 0.0,
                "Fe{2+}": 0.0,
                "Mn{2+}": 0.0,
                "Ti{4+}": 0.0,
                "Ca{2+}": 0.0,
                "Na{+}": 0.0,
                "K{+}": 1.0,
            }
        )
        em = m.endmembers(cations)
        assert not em.isna().any()


# ==========================================================
# Garnet_TC
# ==========================================================


class TestGarnetTC:
    def test_noxy_ncat(self):
        m = Garnet_TC()
        assert m.noxy == 12
        assert m.needsFe == "Fe3"

    def test_endmembers_runs(self):
        m = Garnet_TC()
        cations = pd.Series(
            {
                "Si{4+}": 3.0,
                "Al{3+}": 1.8,
                "Fe{3+}": 0.2,
                "Mg{2+}": 1.0,
                "Fe{2+}": 1.0,
                "Mn{2+}": 0.3,
                "Ca{2+}": 0.7,
            }
        )
        em = m.endmembers(cations)
        assert list(em.index) == ["alm", "py", "spss", "gr", "kho"]
        assert not em.isna().any()

    def test_endmembers_no_fe3(self):
        m = Garnet_TC()
        cations = pd.Series(
            {
                "Si{4+}": 3.0,
                "Al{3+}": 2.0,
                "Fe{3+}": 0.0,
                "Mg{2+}": 1.5,
                "Fe{2+}": 1.5,
                "Mn{2+}": 0.0,
                "Ca{2+}": 0.0,
            }
        )
        em = m.endmembers(cations)
        assert em["py"] == pytest.approx(0.5)
        assert em["alm"] == pytest.approx(0.5)
        assert em["spss"] == pytest.approx(0.0)
        assert em["gr"] == pytest.approx(0.0)
        assert em["kho"] == pytest.approx(0.0)
        assert em.sum() == pytest.approx(1.0)


# ==========================================================
# Garnet_TCnoMn
# ==========================================================


class TestGarnetTCnoMn:
    def test_noxy_ncat(self):
        m = Garnet_TCnoMn()
        assert m.noxy == 12
        assert m.needsFe == "Fe3"

    def test_endmembers_runs(self):
        m = Garnet_TCnoMn()
        cations = pd.Series(
            {
                "Si{4+}": 3.0,
                "Al{3+}": 1.8,
                "Fe{3+}": 0.2,
                "Mg{2+}": 1.0,
                "Fe{2+}": 1.0,
                "Mn{2+}": 0.3,
                "Ca{2+}": 0.7,
            }
        )
        em = m.endmembers(cations)
        assert not em.isna().any()
