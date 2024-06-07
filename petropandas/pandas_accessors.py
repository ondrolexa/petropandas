import importlib.resources
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyparsing
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from periodictable import formula, oxygen

REE = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Sc",
    "Y",
]

germ = importlib.resources.files("petropandas") / "data" / "germ.json"
with open(germ) as fp:
    standards = json.load(fp)


def oxideprops(f):
    ncat, element = f.structure[0]
    noxy = f.atoms[oxygen]
    charge = 2 * noxy // ncat
    return {
        "mass": f.mass,
        "cation": element.ion[charge],
        "noxy": noxy,
        "ncat": ncat,
        "charge": charge,
        "elfrac": f.mass_fraction[element],
    }


def elementprops(f):
    return {
        "mass": f.mass,
    }


class MissingColumns(Exception):
    def __init__(self, cols):
        super().__init__(f"Must have {cols} in columns.")


@pd.api.extensions.register_dataframe_accessor("oxides")
class OxidesAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        # verify there is a oxides column
        valid = []
        self._oxides = []
        self._oxides_props = []
        self._others = []
        for col in obj.columns:
            try:
                f = formula(col)
                if (len(f.atoms) == 2) and (oxygen in f.atoms):
                    valid.append(True)
                    self._oxides.append(col)
                    self._oxides_props.append(oxideprops(f))
                else:
                    valid.append(False)
                    self._others.append(col)
            except (ValueError, pyparsing.exceptions.ParseException):
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("oxides")

    @property
    def props(self):
        return pd.DataFrame(self._oxides_props, index=self._oxides)

    @property
    def _df(self):
        return self._obj[self._oxides]

    def _final(self, df, **kwargs):
        return pd.concat([df, self._obj[kwargs.get("keep", self._others)]], axis=1)

    def df(self, **kwargs):
        return self._final(self._df, **kwargs)

    def scale(self, **kwargs):
        to = kwargs.get("to", 100.0)
        res = to * self._df.div(self._df.sum(axis=1), axis=0)
        return self._final(res, **kwargs)

    def molprop(self, **kwargs):
        res = self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def cat_number(self, **kwargs):
        res = self.props["ncat"] * self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def oxy_number(self, **kwargs):
        res = self.props["noxy"] * self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def onf(self, noxy, **kwargs):
        res = noxy / self.oxy_number.sum(axis=1)
        return self._final(res, **kwargs)

    def cnf(self, ncat, **kwargs):
        res = ncat / self.cat_number.sum(axis=1)
        return self._final(res, **kwargs)

    def cations(self, **kwargs):
        noxy = kwargs.get("noxy", 1)
        ncat = kwargs.get("ncat", 1)
        tocat = kwargs.get("tocat", False)
        if tocat:
            df = self.cat_number.multiply(self.cnf(ncat), axis=0)
            df.columns = [str(cat) for cat in self.props["cation"]]
            return self._final(df, **kwargs)
        else:
            df = self.cat_number.multiply(self.onf(noxy), axis=0)
            df.columns = [str(cat) for cat in self.props["cation"]]
            return self._final(df, **kwargs)


@pd.api.extensions.register_dataframe_accessor("elements")
class ElementsAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        # verify there is an element column
        valid = []
        self._elements = []
        self._elements_props = []
        self._others = []
        for col in obj.columns:
            try:
                f = formula(col)
                if (len(f.atoms) == 1) and (is_numeric_dtype(obj[col].dtype)):
                    valid.append(True)
                    self._elements.append(col)
                    self._elements_props.append(elementprops(f))
                else:
                    valid.append(False)
                    self._others.append(col)
            except (ValueError, pyparsing.exceptions.ParseException):
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("elements")

    @property
    def props(self):
        return pd.DataFrame(self._elements_props, index=self._elements)

    @property
    def _df(self):
        return self._obj[self._elements]

    def _final(self, df, **kwargs):
        return pd.concat([df, self._obj[kwargs.get("keep", self._others)]], axis=1)

    def df(self, **kwargs):
        return self._final(self._df, **kwargs)


@pd.api.extensions.register_dataframe_accessor("ree")
class REEAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        # verify there is a REE column
        valid = []
        self._ree = []
        self._ree_props = []
        self._others = []
        for col in obj.columns:
            if col in REE:
                valid.append(True)
                self._ree.append(col)
                self._ree_props.append(elementprops(formula(col)))
            else:
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("REE")

    @property
    def props(self):
        return pd.DataFrame(self._ree_props, index=self._ree)

    @property
    def _df(self):
        return self._obj[self._ree]

    def _final(self, df, **kwargs):
        select = kwargs.get("select", [])
        if select:
            df = df[select]
        return pd.concat([df, self._obj[kwargs.get("keep", self._others)]], axis=1)

    def df(self, **kwargs):
        return self._final(self._df, **kwargs)

    def normalize(self, **kwargs):
        reservoir = kwargs.get("reservoir", "CI Chondrites")
        reference = kwargs.get("reference", "McDonough & Sun 1995")
        source = kwargs.get("source", reference)
        nrm = pd.Series(standards[reservoir][reference][source])
        res = self._df / nrm
        res = res[self._ree]
        if kwargs.get("Eu", False):
            res["Eu/Eu*"] = res["Eu"] / np.sqrt(res["Sm"] * res["Gd"])
        if kwargs.get("GdYb", False):
            res["Gd/Yb"] = res["Gd"] / res["Yb"]
        return self._final(res, **kwargs)

    def plot(self, **kwargs):
        if "select" not in kwargs:
            kwargs["select"] = [
                "La",
                "Ce",
                "Pr",
                "Nd",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
            ]
        fig, ax = plt.subplots()
        ax.set(yscale="log")
        if kwargs.get("grouped", True):
            ree = self.df(**kwargs).melt(id_vars=kwargs.get("keep", self._others), var_name="ree")
            sns.lineplot(
                x="ree",
                y="value",
                data=ree,
                hue=kwargs.get("hue", None),
                errorbar="ci",
                legend="brief",
                ax=ax,
            )
        else:
            sns.lineplot(data=self._df.T, legend=False, ax=ax)
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    pass
