import pandas as pd
import pytest

from petropandas import ElementsAccessor, OxidesAccessor, REEAccessor  # noqa: F401


@pytest.fixture
def data():
    return pd.DataFrame(
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
        columns=["SiO2", "TiO2", "La", "Gd", "Lu", "F"],
    )


def test_OxidesAccessor(data):
    assert all(data.oxides._df.columns == ["SiO2", "TiO2"])


def test_ElementsAccessor(data):
    assert all(data.elements._df.columns == ["La", "Gd", "Lu", "F"])


def test_REEAccessor(data):
    assert all(data.ree._df.columns == ["La", "Gd", "Lu"])
