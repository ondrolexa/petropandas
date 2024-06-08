import importlib.resources
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyparsing
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from periodictable import formula, oxygen

from petropandas.constants import (
    AGECOLS,
    COLNAMES,
    ISOPLOT,
    ISOPLOT_FORMATS,
    REE,
    REE_PLOT,
)

germ = importlib.resources.files("petropandas") / "data" / "germ.json"
with open(germ) as fp:
    standards = json.load(fp)


ppconfig = {"isoplot_default_format": 2, "colnames": COLNAMES}


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
    def __init__(self, col):
        super().__init__(f"Must have {col} in columns.")


class NotTextualIndex(Exception):
    def __init__(self):
        super().__init__("Index is not textual.")


class NotTextualColumn(Exception):
    def __init__(self, col):
        super().__init__(f"Column {col} is not textual.")


class TemplateNotDefined(Exception):
    def __init__(self, tmpl):
        super().__init__(f"Column definition {tmpl} is not defined. Check `ppconfig['colnames']`")


@pd.api.extensions.register_dataframe_accessor("petro")
class PetroAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def search(self, s, on=None) -> pd.DataFrame:
        """Select subset of data from dataframe containing string s in index or column.

        Note: Works only with non-numeric index or column

        Args:
            s (str): Returns all rows which contain string s in index or column.
            on (str or None): Name of column used for search. When `None` the index is used

        Returns:
            Dataframe with selected data
        """
        if on is None:
            col = self._obj.index
        else:
            if on not in self._obj:
                raise MissingColumns(on)
            col = self._obj[on]
        if not is_numeric_dtype(col):
            col = pd.Series([str(v) for v in col], index=self._obj.index)
            return self._obj.loc[col.str.contains(s)].copy()
        else:
            if on is None:
                raise NotTextualIndex()
            else:
                raise NotTextualColumn(on)

    def fix_columns(self, template) -> pd.DataFrame:
        """Rename columns according to predefined template.

        Check `ppconfig['colnames']` for available templates. User-defined templates
        could be added. Template is a dict used for `pandas.DataFrame.rename`.

        Args:
            template (str): Name of renaming template

        Returns:
            Dataframe with renamed columns
        """
        if template not in ppconfig["colnames"]:
            raise TemplateNotDefined(template)
        return self._obj.rename(columns=ppconfig["colnames"][template])

    def strip_columns(self) -> pd.DataFrame:
        """Strip whitespaces from column names

        Returns:
            Dataframe with stripped column names
        """
        return self._obj.rename(columns=lambda x: x.strip())

    def calc(self, expr, name=None) -> pd.DataFrame:
        """Calculate a new column using expression

        Evaluate a string describing operations on DataFrame columns.

        Args:
            expr (str): The expression string to evaluate.
            name (str): Name of column to store result. When `None` the expression
                is used as name.

        Returns:
            Dataframe with calculated column
        """
        self._obj[expr] = self._obj.eval(expr)
        return self._obj


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
    def props(self) -> pd.DataFrame:
        """Returns properties of oxides in data."""
        return pd.DataFrame(self._oxides_props, index=self._oxides)

    @property
    def _df(self) -> pd.DataFrame:
        """Returns dataframe with only oxides in columns."""
        return self._obj[self._oxides]

    def _final(self, df, **kwargs):
        select = kwargs.get("select", [])
        if select:
            df = df[select]
        return pd.concat([df, self._obj[kwargs.get("keep", self._others)]], axis=1)

    def df(self, **kwargs) -> pd.DataFrame:
        """Returns dataframe.

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Dataframe with oxides and additional columns
        """
        return self._final(self._df, **kwargs)

    def mean(self, **kwargs) -> pd.DataFrame:
        """Return Dataframe with single row of arithmetic means of oxide columns"""
        return pd.DataFrame(self._df.mean(axis=0)).T

    def scale(self, **kwargs) -> pd.DataFrame:
        """Normalize oxide values to given sum.

        Keyword Args:
            to (float): Sum of oxides. Default 100.0
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Scaled dataframe
        """
        to = kwargs.get("to", 100.0)
        res = to * self._df.div(self._df.sum(axis=1), axis=0)
        return self._final(res, **kwargs)

    def molprop(self, **kwargs) -> pd.DataFrame:
        """Convert oxides weight percents to molar proportions.

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Dataframe with molar proportions

        """
        res = self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def cat_number(self, **kwargs) -> pd.DataFrame:
        """Calculate cations number.

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Dataframe with molar proportions

        """
        res = self.props["ncat"] * self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def oxy_number(self, **kwargs) -> pd.DataFrame:
        """Calculate oxugens number.

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Dataframe with molar proportions

        """
        res = self.props["noxy"] * self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def onf(self, noxy) -> pd.Series:
        """Oxygen normalisation factor - ideal oxygens / sum of oxygens

        Args:
            noxy (int): ideal oxygens

        Returns:
            pandas.Series: oxygen normalisation factors

        """
        return noxy / self.oxy_number(keep=[]).sum(axis=1)

    def cnf(self, ncat) -> pd.Series:
        """Cation normalisation factor - ideal cations / sum of cations

        Args:
            ncat (int): ideal cations

        Returns:
            pandas.Series: cation normalisation factors

        """
        return ncat / self.cat_number(keep=[]).sum(axis=1)

    def cations(self, **kwargs) -> pd.DataFrame:
        """Cations calculated on the basis of oxygens or cations.

        Keyword Args:
            noxy (int, optional): ideal number of oxygens. Default 1
            ncat (int, optional): ideal number of cations. Default 1
            tocat (bool, optional): when ``True`` normalized to ``ncat``,
                otherwise to ``noxy``. Default ``False``

        Returns:
            Dataframe with calculated cations

        """
        noxy = kwargs.get("noxy", 1)
        ncat = kwargs.get("ncat", 1)
        tocat = kwargs.get("tocat", False)
        if tocat:
            df = self.cat_number(keep=[]).multiply(self.cnf(ncat), axis=0)
            df.columns = [str(cat) for cat in self.props["cation"]]
            return self._final(df, **kwargs)
        else:
            df = self.cat_number(keep=[]).multiply(self.onf(noxy), axis=0)
            df.columns = [str(cat) for cat in self.props["cation"]]
            return self._final(df, **kwargs)

    def charges(self, ncat, **kwargs) -> pd.DataFrame:
        """Calculates charges based on number of cations.

        Args:
            ncat (int): number of cations

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Dataframe with charges

        """
        charge = self.cat_number(keep=[]).mul(self.cnf(ncat), axis=0) * self.props["charge"]
        return self._final(charge, **kwargs)

    def apatite_correction(self, **kwargs) -> pd.DataFrame:
        """Apatite correction

        Note:
            All P2O5 is assumed to be apatite based and is removed from composition

                CaO mol% = CaO mol% - (10 / 3) * P2O5 mol%

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Apatite corrected dataframe

        """
        if ("P2O5" in self._oxides) and ("CaO" in self._oxides):
            df = self._df.div(self.props["mass"])
            df = df.div(df.sum(axis=1), axis=0)
            df["CaO"] = (df["CaO"] - (10 / 3) * df["P2O5"]).clip(lower=0)
            df = df.mul(self.props["mass"], axis=1)
            df = df.drop(columns="P2O5")
            df = df.div(df.sum(axis=1), axis=0).mul(self._df.sum(axis=1), axis=0)
            return self._final(df, **kwargs)
        else:
            print("Both CaO and P2O5 not in data. Nothing changed.")
            return self._final(self._df, **kwargs)

    def convert_Fe(self, **kwargs) -> pd.DataFrame:
        """Recalculate FeO to Fe2O3 or vice-versa.

        Note:
            When only FeO exists, all is recalculated to Fe2O3. When only Fe2O3
            exists, all is recalculated to FeO. When both exists, Fe2O3 is
            recalculated and added to FeO. Otherwise datatable is not changed.

        Keyword Args:
            to (str): to what iron oxide Fe should be converted. Default `"FeO"`
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Dataframe with converted Fe oxide

        """
        to = kwargs.get("to", "FeO")
        if (to == "FeO") and ("Fe2O3" in self._oxides):
            Fe3to2 = 2 * formula("FeO").mass / formula("Fe2O3").mass
            res = self._df.copy()
            if "FeO" in self._oxides:
                res["FeO"] += Fe3to2 * res["Fe2O3"]
            else:
                res["FeO"] = Fe3to2 * res["Fe2O3"]
            res = res.drop(columns="Fe2O3")
            return self._final(res, **kwargs)
        elif (to == "Fe2O3") and ("FeO" in self._oxides):
            Fe2to3 = formula("Fe2O3").mass / formula("FeO").mass / 2
            res = self._df.copy()
            if "Fe2O3" in self._oxides:
                res["Fe2O3"] += Fe2to3 * res["FeO"]
            else:
                res["Fe2O3"] = Fe2to3 * res["FeO"]
            res = res.drop(columns="FeO")
            return self._final(res, **kwargs)
        else:
            print("Both FeO and Fe2O3 not in data. Nothing changed.")
            return self._final(self._df, **kwargs)

    def recalculate_Fe(self, noxy, ncat, **kwargs) -> pd.DataFrame:
        """Recalculate Fe based on charge balance.

        Note:
            Either both FeO and Fe2O3 are present or any of then, the composition
            is modified to fullfil charge balance for given cations and oxygens.

        Args:
            noxy (int): ideal number of oxygens. Default 1
            ncat (int): ideal number of cations. Default 1

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default all columns.

        Returns:
            Dataframe with recalculated Fe

        """
        charge = self.cat_number(keep=[]).mul(self.cnf(ncat), axis=0)
        if ("Fe2O3" in self._oxides) & ("FeO" not in self._oxides):
            charge["Fe2O3"].loc[pd.isna(self._df["Fe2O3"])] = 0
            chargedef = 2 * noxy - self.charges(ncat, keep=[]).sum(axis=1)
            toconv = chargedef
            charge["Fe2O3"] += toconv
            charge["FeO"] = -toconv
            ncats = self.props["ncat"]
            ncats["FeO"] = 1
            mws = self.props["mass"]
            mws["FeO"] = formula("FeO").mass
        elif "Fe2O3" in self._oxides:
            charge["Fe2O3"].loc[pd.isna(self._df["Fe2O3"])] = 0
            chargedef = 2 * noxy - self.charges(ncat, keep=[]).sum(axis=1)
            toconv = chargedef.clip(lower=0, upper=charge["FeO"])
            charge["Fe2O3"] += toconv
            charge["FeO"] = charge["FeO"] - toconv
            ncats = self.props["ncat"]
            mws = self.props["mass"]
        elif "FeO" in self._oxides:
            chargedef = 2 * noxy - self.charges(ncat, keep=[]).sum(axis=1)
            charge["Fe2O3"] = chargedef.clip(lower=0, upper=charge["FeO"])
            charge["FeO"] = charge["FeO"] - charge["Fe2O3"]
            ncats = self.props["ncat"].copy()
            ncats["Fe2O3"] = 2
            mws = self.props["mass"].copy()
            mws["Fe2O3"] = formula("Fe2O3").mass
        else:
            print("No Fe in data. Nothing changed")
            return self._final(self._df, **kwargs)
        res = self._df.copy()
        ncharge = charge / ncat
        df = ncharge.mul(mws).mul(self.cat_number(keep=[]).sum(axis=1), axis="rows").div(ncats)
        res[df.columns] = df
        return self._final(res, **kwargs)


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
        select = kwargs.get("select", [])
        if select:
            df = df[select]
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
        res["Eu/Eu*"] = res["Eu"] / np.sqrt(res["Sm"] * res["Gd"])
        res["Gd/Yb"] = res["Gd"] / res["Yb"]
        return self._final(res, **kwargs)

    def plot(self, **kwargs):
        if "select" not in kwargs:
            kwargs["select"] = REE_PLOT
        fig, ax = plt.subplots()
        ax.set(yscale="log")
        ree = self.df(**kwargs).melt(
            id_vars=kwargs.get("keep", self._others),
            var_name="ree",
            ignore_index=False,
        )
        if kwargs.get("grouped", False):
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
            sns.lineplot(
                x="ree",
                y="value",
                data=ree,
                hue=kwargs.get("hue", None),
                units=ree.index,
                estimator=None,
                legend="brief",
                ax=ax,
            )
        plt.show()


@pd.api.extensions.register_dataframe_accessor("isoplot")
class IsoplotAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        # verify there is a isoplot column
        valid = []
        self._isoplot = []
        self._others = []
        for col in obj.columns:
            if col in ISOPLOT:
                valid.append(True)
                self._isoplot.append(col)
            else:
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("isoplot")

    @property
    def _df(self):
        return self._obj[self._isoplot]

    def clipboard(self, **kwargs):
        iso = kwargs.get("iso", ppconfig["isoplot_default_format"])
        df = self._df[ISOPLOT_FORMATS[iso]]
        if "C" in kwargs:
            df["C"] = self._obj[kwargs["C"]]
        else:
            df["C"] = None
        if "omit" in kwargs:
            df["omit"] = self._obj[kwargs["omit"]]
        else:
            df["omit"] = None
        if "comment" in kwargs:
            df["comment"] = self._obj[kwargs["comment"]]
        else:
            df["comment"] = None
        df.to_clipboard(header=False, index=False)

    def calc_ages(self, **kwargs):
        iso = kwargs.get("iso", ppconfig["isoplot_default_format"])
        self.clipboard(**kwargs)
        print(f"Data in format {iso} copied to clipboard")
        print("Calc ages with Sracey-Kramers, discordance and digits 3")
        input("Then copy to clipboard and press Enter to continue...")
        ages = pd.read_clipboard(header=None)
        if ages.shape[1] == 9:
            ages.columns = AGECOLS
            ages.index = self._obj.index
            for col in AGECOLS:
                self._obj[col] = ages[col]
            print("Ages added to data")
        else:
            print("Wrong shape of data. Set correct options and try again.")


if __name__ == "__main__":  # pragma: no cover
    pass
