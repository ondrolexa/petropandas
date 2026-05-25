import pytest

from petropandas import config, mindb, pd  # noqa: F401
from petropandas.data import grt_profile, minerals, mnz_sb  # noqa: F401

grt = mindb.Garnet_Fe2()

# ---------- fixtures ----------


@pytest.fixture
def simple():
    return pd.DataFrame(
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
        columns=["SiO2", "TiO2", "La", "Gd", "Lu", "F"],
    )


@pytest.fixture
def oxides_df():
    return pd.DataFrame(
        {
            "SiO2": [45.0, 50.0],
            "Al2O3": [15.0, 20.0],
            "FeO": [10.0, 8.0],
            "MgO": [5.0, 12.0],
            "CaO": [10.0, 5.0],
            "Na2O": [3.0, 2.0],
            "K2O": [2.0, 3.0],
            "Sample": ["A", "B"],
        }
    )


@pytest.fixture
def elements_df():
    return pd.DataFrame(
        {
            "La": [10, 20],
            "Ce": [30, 40],
            "Pr": [5, 6],
            "Sm": [2, 3],
            "Eu": [0.5, 1.0],
            "Gd": [2, 4],
            "Yb": [1, 2],
            "Lu": [0.5, 0.8],
            "SiO2": [45, 50],
            "MgO": [5, 10],
        }
    )


@pytest.fixture
def isoplot_df():
    return pd.DataFrame(
        {
            "38/06": [18.77, 19.91],
            "38/06_Err": [0.45, 0.53],
            "07/06": [0.0583, 0.0535],
            "07/06_Err": [0.0015, 0.0014],
            "RhoXY": [-0.14, -0.05],
        }
    )


@pytest.fixture
def ions_df():
    return pd.DataFrame(
        {
            "Mg{2+}": [0.5, 1.0],
            "Fe{2+}": [0.3, 0.5],
            "Ca{2+}": [0.2, 0.3],
            "SiO2": [45, 50],
        }
    )


@pytest.fixture
def str_index_df():
    return pd.DataFrame(
        {"SiO2": [45.0, 50.0], "Al2O3": [15.0, 20.0]}, index=["p-01", "p-02"]
    )


@pytest.fixture
def multi_col_df():
    return pd.DataFrame(
        {
            "SiO2": [45, 50, 55],
            "Al2O3": [15, 20, 25],
            "MgO": [5, 10, 15],
            "CaO": [10, 8, 6],
        }
    )


# ==========================================================
# PetroAccessor (.petro)
# ==========================================================


class TestPetroAccessor:
    def test_search_on_column(self):
        result = minerals.petro.search("pl-", on="Comment")
        assert len(result) == 3

    def test_search_on_index_textual(self, str_index_df):
        result = str_index_df.petro.search("p-0")
        assert len(result) == 2

    def test_search_regex(self):
        result = minerals.petro.search("p-0[1-5]", on="Comment", regex=True)
        assert len(result) == 0

    def test_search_literal(self):
        result = minerals.petro.search("pl-", on="Comment", regex=False)
        assert len(result) == 3

    def test_search_missing_column_raises(self):
        with pytest.raises(Exception):
            minerals.petro.search("foo", on="NotAColumn")

    def test_search_numeric_index_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        with pytest.raises(Exception):
            df.petro.search("0")

    def test_fix_columns_valid(self):
        df = pd.DataFrame(
            {
                "238U/206Pb": [18.77],
                "238U/206Pb2s": [0.45],
                "207Pb/206Pb": [0.058],
                "207Pb/206Pb2s": [0.001],
                "rho": [-0.14],
            }
        )
        result = df.petro.fix_columns("SB")
        assert "38/06" in result.columns
        assert "RhoXY" in result.columns

    def test_fix_columns_invalid_raises(self):
        with pytest.raises(Exception):
            minerals.petro.fix_columns("NONEXISTENT")

    def test_strip_columns(self):
        df = pd.DataFrame({" SiO2 ": [1.0], " Al2O3 ": [2.0]})
        result = df.petro.strip_columns()
        assert "SiO2" in result.columns
        assert "Al2O3" in result.columns

    def test_calc(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df.petro.calc("a + b", name="c")
        assert "c" in result.columns
        assert result["c"].tolist() == [4, 6]

    def test_calc_uses_expr_as_name(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df.petro.calc("a + b")
        assert "a + b" in result.columns

    def test_to_latex_default(self, simple):
        result = simple.petro.to_latex()
        assert isinstance(result, str)
        assert "SiO2" in result

    def test_to_latex_with_total(self, simple):
        result = simple.petro.to_latex(total=True)
        assert "Total" in result

    def test_to_latex_not_transposed(self, simple):
        result = simple.petro.to_latex(transpose=False)
        assert isinstance(result, str)

    def test_to_latex_precision(self, simple):
        result = simple.petro.to_latex(precision=3)
        assert isinstance(result, str)


# ==========================================================
# IsoplotAccessor (.isoplot)
# ==========================================================


class TestIsoplotAccessor:
    def test_df_default_format(self, isoplot_df):
        result = isoplot_df.isoplot.df()
        assert "38/06" in result.columns
        assert "07/06" in result.columns

    def test_df_with_color(self, isoplot_df):
        result = isoplot_df.isoplot.df(C="RhoXY")
        assert "C" in result.columns

    def test_df_with_omit(self, isoplot_df):
        result = isoplot_df.isoplot.df(omit="RhoXY")
        assert "omit" in result.columns

    def test_df_with_comment(self, isoplot_df):
        result = isoplot_df.isoplot.df(comment="RhoXY")
        assert "comment" in result.columns

    def test_missing_isoplot_columns_raises(self):
        df = pd.DataFrame({"SiO2": [1.0]})
        with pytest.raises(Exception):
            _ = df.isoplot

    def test_df_with_isoplot_index(self, isoplot_df):
        df = isoplot_df.copy()
        df["07/06_Err"] = [0.001, 0.002]
        _ = df.isoplot


# ==========================================================
# AccessorTemplate (base) / OxidesAccessor (.oxides)
# ==========================================================


class TestOxidesAccessor:
    def test_oxides_columns_filtered(self, simple):
        assert all(simple.oxides._df.columns == ["SiO2", "TiO2"])

    def test_missing_oxides_raises(self):
        df = pd.DataFrame({"La": [1.0], "Ce": [2.0]})
        with pytest.raises(Exception):
            _ = df.oxides

    def test_molprop(self, oxides_df):
        result = oxides_df.oxides.molprop()
        assert all(result > 0) or all(result.min() > 0)

    def test_props(self, oxides_df):
        props = oxides_df.oxides.props
        assert "mass" in props.columns
        assert "cation" in props.columns
        assert "noxy" in props.columns
        assert "ncat" in props.columns

    def test_df(self, oxides_df):
        result = oxides_df.oxides.df()
        assert list(result.columns) == [
            "SiO2",
            "Al2O3",
            "FeO",
            "MgO",
            "CaO",
            "Na2O",
            "K2O",
        ]

    def test_df_keep(self, oxides_df):
        result = oxides_df.oxides.df(keep=["Sample"])
        assert "Sample" in result.columns

    def test_df_keep_all(self, oxides_df):
        result = oxides_df.oxides.df(keep="all")
        assert "Sample" in result.columns

    def test_dropna(self, oxides_df):
        result = oxides_df.oxides.dropna()
        assert not result.empty

    def test_mean(self, oxides_df):
        result = oxides_df.oxides.mean()
        assert result.shape == (1, 7)

    def test_sum(self, oxides_df):
        result = oxides_df.oxides.sum()
        assert result.shape == (1, 7)

    def test_scale(self, oxides_df):
        result = oxides_df.oxides.scale(to=100)
        assert result.sum(axis=1).iloc[0] == pytest.approx(100, abs=0.01)

    def test_oxwt(self, oxides_df):
        result = oxides_df.oxides.oxwt()
        assert list(result.columns) == [
            "SiO2",
            "Al2O3",
            "FeO",
            "MgO",
            "CaO",
            "Na2O",
            "K2O",
        ]

    def test_cat_number(self, oxides_df):
        result = oxides_df.oxides.cat_number()
        assert all(result > 0)

    def test_oxy_number(self, oxides_df):
        result = oxides_df.oxides.oxy_number()
        assert all(result > 0)

    def test_onf(self, oxides_df):
        result = oxides_df.oxides.onf(12)
        assert len(result) == len(oxides_df)

    def test_cnf(self, oxides_df):
        result = oxides_df.oxides.cnf(8)
        assert len(result) == len(oxides_df)

    def test_cations_default(self, oxides_df):
        result = oxides_df.oxides.cations()
        assert len(result) == len(oxides_df)

    def test_cations_tocat(self, oxides_df):
        result = oxides_df.oxides.cations(tocat=True)
        assert len(result) == len(oxides_df)

    def test_cations_with_noxy(self, oxides_df):
        result = oxides_df.oxides.cations(noxy=12)
        assert len(result) == len(oxides_df)

    def test_charges(self, oxides_df):
        result = oxides_df.oxides.charges(8)
        assert len(result) == len(oxides_df)

    def test_charge_def(self, oxides_df):
        result = oxides_df.oxides.charge_def(ncat=8, noxy=12)
        assert len(result) == len(oxides_df)

    def test_charge_def_with_mineral(self, oxides_df):
        result = oxides_df.oxides.charge_def(mineral=grt)
        assert len(result) == len(oxides_df)

    def test_omega(self, oxides_df):
        result = oxides_df.oxides.omega(ncat=8, noxy=12)
        assert len(result) == len(oxides_df)

    def test_omega_with_mineral(self, oxides_df):
        result = oxides_df.oxides.omega(mineral=grt)
        assert len(result) == len(oxides_df)

    def test_apatite_correction_with_p_and_ca(self):
        df = pd.DataFrame({"SiO2": [45.0], "CaO": [10.0], "P2O5": [1.0], "FeO": [5.0]})
        result = df.oxides.apatite_correction()
        assert "P2O5" not in result.columns
        assert "CaO" in result.columns

    def test_apatite_correction_no_p(self):
        df = pd.DataFrame({"SiO2": [45.0], "CaO": [10.0], "FeO": [5.0]})
        result = df.oxides.apatite_correction()
        assert "CaO" in result.columns

    def test_convert_Fe_to_FeO(self):
        df = pd.DataFrame({"SiO2": [45.0], "Fe2O3": [5.0], "MgO": [10.0]})
        result = df.oxides.convert_Fe(to="FeO")
        assert "FeO" in result.columns
        assert "Fe2O3" not in result.columns
        assert result["FeO"].iloc[0] > 0

    def test_convert_Fe_to_Fe2O3(self):
        df = pd.DataFrame({"SiO2": [45.0], "FeO": [5.0], "MgO": [10.0]})
        result = df.oxides.convert_Fe(to="Fe2O3")
        assert "Fe2O3" in result.columns
        assert "FeO" not in result.columns

    def test_convert_Fe_both_to_feo(self):
        df = pd.DataFrame({"SiO2": [45.0], "FeO": [3.0], "Fe2O3": [5.0], "MgO": [10.0]})
        result = df.oxides.convert_Fe(to="FeO")
        assert "FeO" in result.columns
        assert "Fe2O3" not in result.columns
        assert result["FeO"].iloc[0] > 3.0

    def test_convert_Fe_no_iron_returns_unchanged(self):
        df = pd.DataFrame({"SiO2": [45.0], "MgO": [10.0]})
        result = df.oxides.convert_Fe(to="FeO")
        assert list(result.columns) == ["SiO2", "MgO"]

    def test_recalculate_Fe_fe2o3_only(self):
        df = pd.DataFrame({"SiO2": [45.0], "Fe2O3": [5.0], "MgO": [10.0]})
        result = df.oxides.recalculate_Fe(ncat=8, noxy=12)
        assert "FeO" in result.columns

    def test_recalculate_Fe_both_feoxides(self):
        df = pd.DataFrame({"SiO2": [45.0], "FeO": [5.0], "Fe2O3": [2.0], "MgO": [10.0]})
        result = df.oxides.recalculate_Fe(ncat=8, noxy=12)
        assert "FeO" in result.columns
        assert "Fe2O3" in result.columns

    def test_recalculate_Fe_feo_only(self):
        df = pd.DataFrame({"SiO2": [45.0], "FeO": [5.0], "MgO": [10.0]})
        result = df.oxides.recalculate_Fe(ncat=8, noxy=12)
        assert "Fe2O3" in result.columns

    def test_recalculate_Fe_no_iron(self):
        df = pd.DataFrame({"SiO2": [45.0], "MgO": [10.0]})
        result = df.oxides.recalculate_Fe(ncat=8, noxy=12)
        assert list(result.columns) == ["SiO2", "MgO"]

    def test_recalculate_Fe_with_mineral(self):
        df = pd.DataFrame({"SiO2": [45.0], "FeO": [5.0], "MgO": [10.0]})
        result = df.oxides.recalculate_Fe(mineral=grt)
        assert "Fe2O3" in result.columns

    def test_endmembers_with_data(self):
        result = grt_profile.oxides.endmembers(grt)
        assert result.shape == (99, 4)
        assert list(result.columns) == ["Alm", "Prp", "Sps", "Grs"]

    def test_endmembers_sum_to_nrows(self):
        result = grt_profile.oxides.endmembers(grt)
        assert result.sum(axis=1).sum() == pytest.approx(len(grt_profile))

    def test_endmembers_keep(self):
        result = grt_profile.oxides.endmembers(grt, keep=["Label"])
        assert result.shape == (99, 5)

    def test_check_stechiometry(self):
        result = grt_profile.oxides.check_stechiometry(grt)
        assert result.name == "misfit"
        assert result.sum() == pytest.approx(0.0008756712880324041)

    def test_apfu(self):
        result = grt_profile.oxides.apfu(grt)
        assert len(result) == len(grt_profile)

    def test_apfu_force(self):
        result = grt_profile.oxides.apfu(grt, force=True)
        assert len(result) == len(grt_profile)

    def test_apfu_feldspar(self):
        fsp = mindb.Feldspar()
        df = pd.DataFrame(
            {"SiO2": [65.0], "Al2O3": [20.0], "CaO": [2.0], "Na2O": [5.0], "K2O": [5.0]}
        )
        result = df.oxides.apfu(fsp)
        assert not result.empty

    def test_endmembers_feldspar(self):
        fsp = mindb.Feldspar()
        df = pd.DataFrame(
            {"SiO2": [65.0], "Al2O3": [20.0], "CaO": [2.0], "Na2O": [5.0], "K2O": [5.0]}
        )
        result = df.oxides.endmembers(fsp)
        assert result.sum(axis=1).iloc[0] == pytest.approx(1.0)

    def test_endmembers_pyroxene_fe2(self):
        px = mindb.Pyroxene_Fe2()
        df = pd.DataFrame(
            {
                "SiO2": [55.0],
                "Al2O3": [2.0],
                "FeO": [10.0],
                "MgO": [20.0],
                "CaO": [10.0],
                "MnO": [0.5],
            }
        )
        result = df.oxides.endmembers(px)
        assert result.sum(axis=1).iloc[0] == pytest.approx(1.0)

    def test_TCbulk_dataframe(self):
        result = grt_profile.oxides.TCbulk(dataframe=True)
        assert "H2O" in result.columns
        assert "SiO2" in result.columns

    def test_TCbulk_with_H2O(self):
        df = pd.DataFrame(
            {
                "SiO2": [50.0],
                "Al2O3": [15.0],
                "CaO": [10.0],
                "MgO": [5.0],
                "FeO": [10.0],
                "K2O": [2.0],
                "Na2O": [3.0],
                "TiO2": [1.0],
                "MnO": [0.5],
            }
        )
        result = df.oxides.TCbulk(H2O=1.0, dataframe=True)
        assert "H2O" in result.columns
        assert result["H2O"].iloc[0] > 0

    def test_TCbulk_H2O_auto(self):
        df = pd.DataFrame(
            {
                "SiO2": [50.0],
                "Al2O3": [15.0],
                "CaO": [10.0],
                "MgO": [5.0],
                "FeO": [10.0],
                "K2O": [2.0],
                "Na2O": [3.0],
                "TiO2": [1.0],
                "MnO": [0.5],
            }
        )
        result = df.oxides.TCbulk(H2O=-1, dataframe=True)
        assert "H2O" in result.columns

    def test_Perplexbulk_dataframe(self):
        result = grt_profile.oxides.Perplexbulk(dataframe=True)
        assert "H2O" in result.columns

    def test_MAGEMin_dataframe(self):
        result = grt_profile.oxides.MAGEMin(dataframe=True)
        assert "SiO2" in result.columns

    def test_MAGEMin_wt_system(self):
        result = grt_profile.oxides.MAGEMin(sys_in="wt", dataframe=True)
        assert "SiO2" in result.columns

    def test_endmembers_no_endmembers_raises(self):
        class FakeMineral:
            has_endmembers = False

        with pytest.raises(Exception):
            grt_profile.oxides.endmembers(FakeMineral())

    def test_sort_oxides(self):
        config["sort_oxides"] = True
        df = pd.DataFrame(
            {"FeO": [10.0], "SiO2": [50.0], "Al2O3": [15.0], "MgO": [5.0]}
        )
        result = df.oxides._df
        assert list(result.columns[:4]) == ["SiO2", "Al2O3", "FeO", "MgO"]
        config["sort_oxides"] = False

    def test_oxides_order_config_sort(self):
        config["sort_oxides"] = True
        df = pd.DataFrame({"PbO": [0.1], "ZrO2": [0.2], "ZnO": [0.3], "Y2O3": [0.4]})
        result = df.oxides._df
        assert not result.empty
        config["sort_oxides"] = False

    def test_endmembers_tocat_false(self):
        result = grt_profile.oxides.endmembers(grt, tocat=False)
        assert result is not None


# ==========================================================
# ElementsAccessor (.elements)
# ==========================================================


class TestElementsAccessor:
    def test_elements_columns_filtered(self, simple):
        assert all(simple.elements._df.columns == ["La", "Gd", "Lu", "F"])

    def test_missing_elements_raises(self):
        df = pd.DataFrame({"SiO2": [1.0], "Al2O3": [2.0]})
        with pytest.raises(Exception):
            _ = df.elements

    def test_molprop(self, elements_df):
        result = elements_df.elements.molprop()
        assert not result.empty

    def test_mean(self, elements_df):
        result = elements_df.elements.mean()
        assert result.shape == (1, 8)

    def test_scale(self, elements_df):
        result = elements_df.elements.scale(to=1.0)
        assert result.sum(axis=1).iloc[0] == pytest.approx(1.0)


# ==========================================================
# REEAccessor (.ree)
# ==========================================================


class TestREEAccessor:
    def test_ree_columns_filtered(self, simple):
        assert all(simple.ree._df.columns == ["La", "Gd", "Lu"])

    def test_missing_ree_raises(self):
        df = pd.DataFrame({"SiO2": [1.0], "Al2O3": [2.0]})
        with pytest.raises(Exception):
            _ = df.ree

    def test_normalize_default(self, elements_df):
        result = elements_df.ree.normalize()
        assert "La" in result.columns
        assert "Eu/Eu*" in result.columns
        assert "Gd/Yb" in result.columns

    def test_normalize_with_reservoir(self, elements_df):
        result = elements_df.ree.normalize(
            reservoir="CI Chondrites", reference="McDonough & Sun 1995"
        )
        assert "Eu/Eu*" in result.columns

    def test_normalize_keep(self, elements_df):
        result = elements_df.ree.normalize(keep=["SiO2"])
        assert "SiO2" in result.columns

    def test_normalize_dropna_false(self, elements_df):
        result = elements_df.ree.normalize(dropna=False)
        assert not result.empty


# ==========================================================
# IonsAccessor (.ions)
# ==========================================================


class TestIonsAccessor:
    def test_ions_validate(self, ions_df):
        assert hasattr(ions_df, "ions")

    def test_missing_ions_raises(self):
        df = pd.DataFrame({"SiO2": [1.0]})
        with pytest.raises(Exception):
            _ = df.ions

    def test_wt(self, ions_df):
        result = ions_df.ions.wt()
        assert "MgO" in result.columns
        assert "FeO" in result.columns
        assert "CaO" in result.columns

    def test_wt_with_omega(self, ions_df):
        result = ions_df.ions.wt(omega=2)
        assert not result.empty

    def test_wt_values_positive(self, ions_df):
        result = ions_df.ions.wt()
        assert (result > 0).all().all()
