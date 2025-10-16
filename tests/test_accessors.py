import pytest

from petropandas import mindb, pd  # noqa: F401
from petropandas.data import grt_profile, minerals  # noqa: F401

grt = mindb.Garnet_Fe2()


@pytest.fixture
def simple():
    return pd.DataFrame(
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
        columns=["SiO2", "TiO2", "La", "Gd", "Lu", "F"],
    )


def test_OxidesAccessor(simple):
    assert all(simple.oxides._df.columns == ["SiO2", "TiO2"])


def test_ElementsAccessor(simple):
    assert all(simple.elements._df.columns == ["La", "Gd", "Lu", "F"])


def test_REEAccessor(simple):
    assert all(simple.ree._df.columns == ["La", "Gd", "Lu"])


def test_endmembers():
    assert pytest.approx(grt_profile.oxides.endmembers(grt).sum().sum()) == len(
        grt_profile
    )


def test_endmembers_keep():
    assert grt_profile.oxides.endmembers(grt, keep=["Label"]).shape == (99, 5)


def test_check_stechiometry():
    assert (
        pytest.approx(grt_profile.oxides.check_stechiometry(grt).sum())
        == 0.0008756712880324041
    )


def test_search():
    assert len(minerals.petro.search("pl-", on="Comment")) == 3
