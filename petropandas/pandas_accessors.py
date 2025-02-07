import importlib.resources
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyparsing
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from periodictable import oxygen
from periodictable.formulas import formula
from periodictable.core import iselement, ision

from petropandas.constants import (
    AGECOLS,
    COLNAMES,
    ISOPLOT,
    ISOPLOT_FORMATS,
    REE,
    REE_PLOT,
)
from petropandas.minerals import Mineral

pp_config = {
    "isoplot_default_format": 2,
    "colnames": COLNAMES,
    "agecols": AGECOLS,
    "isoplot_formats": ISOPLOT_FORMATS,
    "ree_plot": REE_PLOT,
}

germ = importlib.resources.files("petropandas").joinpath("data", "germ.json")
with germ.open() as fp:
    pp_config["reservoirs"] = json.load(fp)


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
        "charge": f.charge,
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
        super().__init__(
            f"Column definition {tmpl} is not defined. Check `pp_config['colnames']`"
        )


class NoEndMembers(Exception):
    def __init__(self, mineral):
        super().__init__(f"Mineral {mineral} has no endmembers method defined.")


@pd.api.extensions.register_dataframe_accessor("petro")
class PetroAccessor:
    """Use `.petro` pandas dataframe accessor."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def search(self, s, on=None, regex=True) -> pd.DataFrame:
        """Select subset of data from dataframe containing string s in index or column.

        Note: Works only with non-numeric index or column

        Args:
            s (str): Returns all rows which contain string s in index or column.
            on (str, optional): Name of column used for search. When `None` the index is used
            regex (bool, optional): If True, assumes the pat is a regular expression. If False,
                treats the pat as a literal string.

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
            return self._obj.loc[col.str.contains(s, regex=regex).fillna(False)].copy()
        else:
            if on is None:
                raise NotTextualIndex()
            else:
                raise NotTextualColumn(on)

    def fix_columns(self, template) -> pd.DataFrame:
        """Rename columns according to predefined template.

        Check `pp_config['colnames']` for available templates. User-defined templates
        could be added. Template is a dict used for `pandas.DataFrame.rename`.

        Args:
            template (str): Name of renaming template

        Returns:
            Dataframe with renamed columns
        """
        if template not in pp_config["colnames"]:
            raise TemplateNotDefined(template)
        return self._obj.rename(columns=pp_config["colnames"][template])

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
    """Use `.oxides` pandas dataframe accessor."""

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
        return pd.DataFrame(self._oxides_props, index=pd.Index(self._oxides))

    @property
    def _df(self) -> pd.DataFrame:
        """Returns dataframe with only oxides in columns."""
        return self._obj[self._oxides]

    def _final(self, df, **kwargs):
        select = kwargs.get("select", [])
        if select:
            df = df[df.columns.intersection(select)]
            rest = df.columns.symmetric_difference(select).difference(df.columns)
            df[rest] = np.nan
        keep = kwargs.get("keep", [])
        if keep == "all":
            keep = self._others
        return pd.concat([df, self._obj[keep]], axis=1)

    def df(self, **kwargs) -> pd.DataFrame:
        """Returns dataframe.

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default [].

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
            keep (list): list of additional columns to be included. Default [].

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
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with molar proportions

        """
        res = self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def cat_number(self, **kwargs) -> pd.DataFrame:
        """Calculate cations number.

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with molar proportions

        """
        res = self.props["ncat"] * self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def oxy_number(self, **kwargs) -> pd.DataFrame:
        """Calculate oxugens number.

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default [].

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
        return noxy / self.oxy_number().sum(axis=1)

    def cnf(self, ncat) -> pd.Series:
        """Cation normalisation factor - ideal cations / sum of cations

        Args:
            ncat (int): ideal cations

        Returns:
            pandas.Series: cation normalisation factors

        """
        return ncat / self.cat_number().sum(axis=1)

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
            df = self.cat_number().multiply(self.cnf(ncat), axis=0)
            df.columns = [str(cat) for cat in self.props["cation"]]
            return self._final(df, **kwargs)
        else:
            df = self.cat_number().multiply(self.onf(noxy), axis=0)
            df.columns = [str(cat) for cat in self.props["cation"]]
            return self._final(df, **kwargs)

    def charges(self, ncat, **kwargs) -> pd.DataFrame:
        """Calculates charges based on number of cations.

        Args:
            ncat (int): number of cations

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included.Default [].

        Returns:
            Dataframe with charges

        """
        charge = self.cat_number().mul(self.cnf(ncat), axis=0) * self.props["charge"]
        return self._final(charge, **kwargs)

    def apatite_correction(self, **kwargs) -> pd.DataFrame:
        """Apatite correction

        Note:
            All P2O5 is assumed to be apatite based and is removed from composition

                CaO mol% = CaO mol% - (10 / 3) * P2O5 mol%

        Keyword Args:
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default [].

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
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with converted Fe oxide

        """
        to = kwargs.get("to", "FeO")
        if to == "FeO":
            if "Fe2O3" in self._oxides:
                Fe3to2 = 2 * formula("FeO").mass / formula("Fe2O3").mass
                res = self._df.copy()
                if "FeO" in self._oxides:
                    res["FeO"] += Fe3to2 * res["Fe2O3"]
                else:
                    res["FeO"] = Fe3to2 * res["Fe2O3"]
                res = res.drop(columns="Fe2O3")
                return self._final(res, **kwargs)
            else:
                return self._final(self._df, **kwargs)
        elif to == "Fe2O3":
            if "FeO" in self._oxides:
                Fe2to3 = formula("Fe2O3").mass / formula("FeO").mass / 2
                res = self._df.copy()
                if "Fe2O3" in self._oxides:
                    res["Fe2O3"] += Fe2to3 * res["FeO"]
                else:
                    res["Fe2O3"] = Fe2to3 * res["FeO"]
                res = res.drop(columns="FeO")
                return self._final(res, **kwargs)
            else:
                return self._final(self._df, **kwargs)
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
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with recalculated Fe

        """
        charge = self.cat_number().mul(self.cnf(ncat), axis=0)
        if ("Fe2O3" in self._oxides) & ("FeO" not in self._oxides):
            charge["Fe2O3"].loc[pd.isna(self._df["Fe2O3"])] = 0
            chargedef = 2 * noxy - self.charges(ncat).sum(axis=1)
            toconv = chargedef
            charge["Fe2O3"] += toconv
            charge["FeO"] = -toconv
            ncats = self.props["ncat"]
            ncats["FeO"] = 1
            mws = self.props["mass"]
            mws["FeO"] = formula("FeO").mass
        elif "Fe2O3" in self._oxides:
            charge["Fe2O3"].loc[pd.isna(self._df["Fe2O3"])] = 0
            chargedef = 2 * noxy - self.charges(ncat).sum(axis=1)
            toconv = chargedef.clip(lower=0, upper=charge["FeO"])
            charge["Fe2O3"] += toconv
            charge["FeO"] = charge["FeO"] - toconv
            ncats = self.props["ncat"]
            mws = self.props["mass"]
        elif "FeO" in self._oxides:
            chargedef = 2 * noxy - self.charges(ncat).sum(axis=1)
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
        df = ncharge.mul(mws).mul(self.cat_number().sum(axis=1), axis="rows").div(ncats)
        res[df.columns] = df
        return self._final(res, **kwargs)

    def endmembers(self, mineral: Mineral, **kwargs) -> pd.DataFrame:
        """Calculate endmembers proportions

        Args:
            mineral: Mineral instance (see `petropandas.minerals`)

        Keyword Args:
            force (bool, optional): when True, remaining cations are added to last site
            select (list): list of oxides to be included. Default all oxides.
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with calculated endmembers

        """
        force = kwargs.get("force", False)
        if mineral.has_endmembers:
            if mineral.needsFe == "Fe2":
                dt = self.convert_Fe()
            elif mineral.needsFe == "Fe3":
                dt = self.recalculate_Fe(mineral.noxy, mineral.ncat)
            else:
                dt = self.df()
            cations = dt.oxides.cations(noxy=mineral.noxy, ncat=mineral.ncat, **kwargs)
            res = []
            for _, row in cations.iterrows():
                res.append(mineral.endmembers(row, force=force))
            return self._final(pd.DataFrame(res, index=self._obj.index), **kwargs)
        else:
            raise NoEndMembers(mineral)

    def TCbulk(self, **kwargs) -> None:
        """Print oxides formatted as THERMOCALC bulk script

        Note:
            The CaO is recalculate using apatite correction based on P205 if available.

        Args:
            H2O (float): wt% of water. When -1 the amount is calculated as 100 - Total
                Default -1.
            oxygen (float): value to calculate moles of ferric iron.
                Moles FeO = FeOtot - 2O and moles Fe2O3 = O. Default 0.01
            system (str): axfile to be used. One of 'MnNCKFMASHTO', 'NCKFMASHTO',
                'KFMASH', 'NCKFMASHTOCr', 'NCKFMASTOCr'. Default 'MnNCKFMASHTO'

        """
        H2O = kwargs.get("H2O", -1)
        oxygen = kwargs.get("oxygen", 0.01)
        system = kwargs.get("system", "MnNCKFMASHTO")
        # fmt: off
        bulk = {
            "MnNCKFMASHTO": ["H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "MnO", "O"],
            "NCKFMASHTO": ["H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O"],
            "KFMASH": ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"],
            "NCKFMASHTOCr": ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3"],
            "NCKFMASTOCr": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "TiO2", "O", "Cr2O3"],
        }
        # fmt: on
        assert system in bulk, "Not valid system"

        df = self.convert_Fe().oxides.apatite_correction()
        # Water
        if "H2O" in bulk[system]:
            if "H2O" not in df:
                if H2O == -1:
                    H2O = 100 - df.sum(axis=1)
                    H2O[H2O < 0] = 0
                else:
                    H2O = H2O * df.sum(axis=1) / (100 - H2O)
                df["H2O"] = H2O
        use = df.columns.intersection(bulk[system])
        df = df[use].oxides.molprop().oxides.scale(to=100 - oxygen)
        if "O" in bulk[system]:
            df["O"] = oxygen
        # add missing
        for lbl in bulk[system]:
            if lbl not in df:
                df[lbl] = 0.0
        print("bulk" + "".join([f"{lbl:>7}" for lbl in bulk[system]]))
        for ix, row in df[bulk[system]].iterrows():
            print("bulk" + "".join([f" {v:6.3f}" for v in row.values]) + f"  % {ix}")

    def Perplexbulk(self, **kwargs) -> None:
        """Print oxides formatted as PerpleX thermodynamic component list

        Note:
            The CaO is recalculate using apatite correction based on P205 if available.

        Args:
            H2O (float): wt% of water. When -1 the amount is calculated as 100 - Total
                Default -1.
            oxygen (float): value to calculate moles of ferric iron.
                Moles FeO = FeOtot - O and moles Fe2O3 = O. Default 0.01
            system (str): axfile to be used. One of 'MnNCKFMASHTO', 'NCKFMASHTO',
                'KFMASH', 'NCKFMASHTOCr', 'NCKFMASTOCr'. Default 'MnNCKFMASHTO'

        """
        H2O = kwargs.get("H2O", -1)
        oxygen = kwargs.get("oxygen", 0.01)
        system = kwargs.get("system", "MnNCKFMASHTO")
        # fmt: off
        bulk = {
            "MnNCKFMASHTO": ["H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "MnO", "O2"],
            "NCKFMASHTO": ["H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O2"],
            "KFMASH": ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"],
            "NCKFMASHTOCr": ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O2", "Cr2O3"],
            "NCKFMASTOCr": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "TiO2", "O2", "Cr2O3"],
        }
        # fmt: on
        assert system in bulk, "Not valid system"

        df = self.convert_Fe().oxides.apatite_correction()
        # Water
        if "H2O" in bulk[system]:
            if "H2O" not in df:
                if H2O == -1:
                    H2O = 100 - df.sum(axis=1)
                    H2O[H2O < 0] = 0
                else:
                    H2O = H2O * df.sum(axis=1) / (100 - H2O)
                df["H2O"] = H2O
        use = df.columns.intersection(bulk[system])
        df = df[use].oxides.molprop().oxides.scale(to=100 - oxygen)
        if "O2" in bulk[system]:
            df["O2"] = 2 * oxygen
        # add missing
        for lbl in bulk[system]:
            if lbl not in df:
                df[lbl] = 0.0
        print("begin thermodynamic component list")
        for ox, val in df[bulk[system]].iloc[0].items():
            print(f"{ox:6s}1 {val:8.5f}      0.00000      0.00000     molar amount")
        print("end thermodynamic component list")

    def MAGEMin(self, **kwargs) -> None:
        """Print oxides formatted as MAGEMin bulk file

        Note:
            The CaO is recalculate using apatite correction based on P205 if available.

        Args:
            H2O (float): wt% of water. When -1 the amount is calculated as 100 - Total
                Default -1.
            oxygen (float): value to calculate moles of ferric iron.
                Moles FeO = FeOtot - 2O and moles Fe2O3 = O. Default 0.01
            db (str): MAGEMin database. 'mp' metapelite (White et al. 2014), 'mb' metabasite
                (Green et al. 2016), 'ig' igneous (Holland et al. 2018), 'um' ultramafic
                (Evans & Frost 2021), 'ume' ultramafic extended (Evans & Frost 2021 + pl, hb and aug
                from Green et al. 2016), 'mpe' Metapelite extended (White et al. 2014,
                Green et al. 2016, Evans & Frost 2021), 'mtl' mantle (Holland et al. 2013).
                Default is "mp"
            sys_in (str): system comp "wt" or "mol". Default is "mol"

        """
        H2O = kwargs.get("H2O", -1)
        oxygen = kwargs.get("oxygen", 0.01)
        db = kwargs.get("db", "mp")
        sys_in = kwargs.get("sys_in", "mol")
        # fmt: off
        bulk = {
            "ig": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "H2O"],
            "mp": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "MnO", "H2O"],
            "mb": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "H2O"],
            "um": ["SiO2", "Al2O3", "MgO", "FeO", "O", "H2O", "S"],
            "ume": ["SiO2", "Al2O3", "MgO", "FeO", "O", "H2O", "S", "CaO", "Na2O"],
            "mpe": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "MnO", "H2O", "CO2", "S"],
            "mtl": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Na2O"],
        }
        # fmt: on
        assert db in bulk, "Not valid database"

        df = self.convert_Fe().oxides.apatite_correction()
        # Water
        if "H2O" in bulk[db]:
            if "H2O" not in df:
                if H2O == -1:
                    H2O = 100 - df.sum(axis=1)
                    H2O[H2O < 0] = 0
                else:
                    H2O = H2O * df.sum(axis=1) / (100 - H2O)
                df["H2O"] = H2O
        use = df.columns.intersection(bulk[db])
        if sys_in == "mol":
            df = df[use].oxides.molprop().oxides.scale(to=100 - oxygen)
        else:
            df = df[use].oxides.scale(to=100 - oxygen)
        if "O" in bulk[db]:
            df["O"] = oxygen
        # add missing
        for lbl in bulk[db]:
            if lbl not in df:
                df[lbl] = 0.0
        print("# HEADER")
        print("title; comments; db; sysUnit; oxide; frac; frac2")
        print("# BULK-ROCK COMPOSITION")
        for ix, row in df[bulk[db]].iterrows():
            oxides = ", ".join(row.keys())
            values = ", ".join([f"{val:.3f}" for val in row.values])
            print(f"{ix};{'petropandas'};{db};{sys_in};[{oxides}];[{values}];")


@pd.api.extensions.register_dataframe_accessor("ions")
class IonsAccessor:
    """Use `.ions` pandas dataframe accessor."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        # verify there is a ions column
        valid = []
        self._ions = []
        self._ions_props = []
        self._others = []
        for col in obj.columns:
            try:
                f = formula(col)
                if (len(f.atoms) == 1) and (is_numeric_dtype(obj[col].dtype)):
                    if ision(next(iter(f.atoms.keys()))):
                        valid.append(True)
                        self._ions.append(col)
                        self._ions_props.append(elementprops(f))
                    else:
                        valid.append(False)
                        self._others.append(col)
                else:
                    valid.append(False)
                    self._others.append(col)
            except (ValueError, pyparsing.exceptions.ParseException):
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("ions")

    @property
    def props(self) -> pd.DataFrame:
        """Returns properties of ions in data."""
        return pd.DataFrame(self._ions_props, index=pd.Index(self._ions))

    @property
    def _df(self) -> pd.DataFrame:
        """Returns dataframe with only ions in columns."""
        return self._obj[self._ions]

    def _final(self, df, **kwargs):
        select = kwargs.get("select", [])
        if select:
            df = df[df.columns.intersection(select)]
            rest = df.columns.symmetric_difference(select).difference(df.columns)
            df[rest] = np.nan
        return pd.concat([df, self._obj[kwargs.get("keep", self._others)]], axis=1)

    def df(self, **kwargs) -> pd.DataFrame:
        """Returns dataframe.

        Keyword Args:
            select (list): list of ions to be included. Default all ions.
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with ions and additional columns
        """
        return self._final(self._df, **kwargs)


@pd.api.extensions.register_dataframe_accessor("elements")
class ElementsAccessor:
    """Use `.elements` pandas dataframe accessor."""

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
                    if iselement(next(iter(f.atoms.keys()))):
                        valid.append(True)
                        self._elements.append(col)
                        self._elements_props.append(elementprops(f))
                    else:
                        valid.append(False)
                        self._others.append(col)
                else:
                    valid.append(False)
                    self._others.append(col)
            except (ValueError, pyparsing.exceptions.ParseException):
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("elements")

    @property
    def props(self) -> pd.DataFrame:
        """Returns properties of elements in data."""
        return pd.DataFrame(self._elements_props, index=pd.Index(self._elements))

    @property
    def _df(self):
        """Returns dataframe with only elements in columns."""
        return self._obj[self._elements]

    def _final(self, df, **kwargs):
        select = kwargs.get("select", [])
        if select:
            df = df[df.columns.intersection(select)]
            rest = df.columns.symmetric_difference(select).difference(df.columns)
            df[rest] = np.nan
        return pd.concat([df, self._obj[kwargs.get("keep", self._others)]], axis=1)

    def df(self, **kwargs) -> pd.DataFrame:
        """Returns dataframe.

        Keyword Args:
            select (list): list of elements to be included. Default all elements.
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with elements and additional columns
        """
        return self._final(self._df, **kwargs)


@pd.api.extensions.register_dataframe_accessor("ree")
class REEAccessor:
    """Use `.ree` pandas dataframe accessor."""

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
    def props(self) -> pd.DataFrame:
        """Returns properties of REE in data."""
        return pd.DataFrame(self._ree_props, index=pd.Index(self._ree))

    @property
    def _df(self):
        """Returns dataframe with only REE in columns."""
        return self._obj[self._ree]

    def _final(self, df, **kwargs):
        select = kwargs.get("select", [])
        if select:
            df = df[df.columns.intersection(select)]
            rest = df.columns.symmetric_difference(select).difference(df.columns)
            df[rest] = np.nan
        return pd.concat([df, self._obj[kwargs.get("keep", self._others)]], axis=1)

    def df(self, **kwargs) -> pd.DataFrame:
        """Returns dataframe.

        Keyword Args:
            select (list): list of elements to be included. Default all elements.
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with elements and additional columns
        """
        return self._final(self._df, **kwargs)

    def normalize(self, **kwargs) -> pd.DataFrame:
        """Normalize REE by reservoir.

        Note:
            Predefined reservoirs are imported from
            [GERM Reservoir Database](https://earthref.org/GERMRD/reservoirs/). You can
            check all available reservoirs in `pp_config["reservoirs"]`.

        Keyword Args:
            reservoir (str): Name of reservoir. Deafult "CI Chondrites"
            reference (str): Reference. Default "McDonough & Sun 1995"
            source (str): Original source. Deafult same as reference.
            select (list): list of elements to be included. Default all elements.
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with normalized REE composition
        """
        reservoir = kwargs.get("reservoir", "CI Chondrites")
        reference = kwargs.get("reference", "McDonough & Sun 1995")
        source = kwargs.get("source", reference)
        nrm = pd.Series(pp_config["reservoirs"][reservoir][reference][source])
        res = self._df / nrm
        res = res[self._ree]
        res["Eu/Eu*"] = res["Eu"] / np.sqrt(res["Sm"] * res["Gd"])
        res["Gd/Yb"] = res["Gd"] / res["Yb"]
        return self._final(res, **kwargs)

    def plot(self, **kwargs):
        """Spiderplot of REE data.

        Note:
            List of REE used for plot could be set in `pp_config["ree_plot"]`

        Keyword Args:
            grouped (bool): When True aggegated data with confidence interval is drawn.
                Default False
            boxplot (bool): When True, boxplot for each REE is drawn. Default False
            boxplot_props (dict): Additional arguments passed to `sns.boxplot`. Default
                `{"color": "grey"}`.
            hue (str or None): Name of columns used for colors.
            palette (string, list, dict, or matplotlib.colors.Colormap): Method for
                choosing the colors to use when mapping the hue semantic.
            legend ("auto", "brief", "full", or False): How to draw the legend.
            title (str): Title of the plot. Default None
            select (list): list of elements to be included. Default all elements.
            filename (str): If not none, plot is saved to file. Default None.
            dpi (int): DPI used for `savefig`. Default 150.
            keep (list): list of additional columns to be included. Default [].
        """
        fig, ax = plt.subplots()
        ax.set(yscale="log")
        ree = self.df(**kwargs).melt(
            id_vars=kwargs.get("keep", self._others),
            var_name="ree",
            ignore_index=False,
        )
        # select only REE for plotting
        ree = ree.loc[ree["ree"].isin(pp_config["ree_plot"])]
        if kwargs.get("grouped", False):
            sns.lineplot(
                data=ree,
                x="ree",
                y="value",
                hue=kwargs.get("hue", None),
                palette=kwargs.get("palette", None),
                errorbar="ci",
                legend=kwargs.get("legend", "brief"),
                ax=ax,
            )
        else:
            sns.lineplot(
                x="ree",
                y="value",
                data=ree,
                hue=kwargs.get("hue", None),
                palette=kwargs.get("palette", None),
                units=ree.index,
                estimator=None,
                legend=kwargs.get("legend", "brief"),
                ax=ax,
            )
        if kwargs.get("boxplot", False):
            sns.boxplot(
                data=ree,
                x="ree",
                y="value",
                flierprops={"ms": 3},
                ax=ax,
                **kwargs.get("boxplot_props", {"color": "grey"}),
            )
        ax.set_title(kwargs.get("title", ""))
        ax.set_xlabel("")
        if "filename" in kwargs:
            fig.tight_layout()
            fig.savefig(
                kwargs.get("filename", "ree_plot.pdf"), dpi=kwargs.get("dpi", 150)
            )
            plt.close(fig)
        else:
            plt.show()


@pd.api.extensions.register_dataframe_accessor("isoplot")
class IsoplotAccessor:
    """Use `.isoplot` pandas dataframe accessor."""

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
        """Returns dataframe with only IsoplotR variables in columns."""
        return self._obj[self._isoplot]

    def clipboard(self, **kwargs):
        """Copy data to clipbord to be used in IsoplotR online.

        Note:
            [IsoplotR online](http://isoplotr.es.ucl.ac.uk/home/index.html)

                Vermeesch, P., 2018, IsoplotR: a free and open toolbox for geochronology.
                Geoscience Frontiers, v.9, p.1479-1493, doi: 10.1016/j.gsf.2018.04.001.

        Keyword Args:
            iso (int): IsoplotR format. Default `pp_config["isoplot_default_format"]`
            C (str): Column to be used as color. Default None
            omit (str): Column to be used as omit. Default None
            comment (str): Column to be used as comment. Default None
        """
        iso = kwargs.get("iso", pp_config["isoplot_default_format"])
        df = self._df[pp_config["isoplot_formats"][iso]]
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

    def calc_ages(self, **kwargs) -> pd.DataFrame | None:
        """Copy data to clipbord, calc ages in IsoplotR online and paste back results.

        Keyword Args:
            iso (int): IsoplotR format. Default `pp_config["isoplot_default_format"]`
            C (str): Column to be used as color. Default None
            omit (str): Column to be used as omit. Default None
            comment (str): Column to be used as comment. Default None

        Returns:
            Dataframe with calculated ages
        """
        iso = kwargs.get("iso", pp_config["isoplot_default_format"])
        self.clipboard(**kwargs)
        print(f"Data in format {iso} copied to clipboard")
        print("Calc ages with Stacey-Kramers, discordance and digits 5")
        input("Then copy to clipboard and press Enter to continue...")
        ages = pd.read_clipboard(header=None)
        if ages.shape[1] == 9:
            ages.columns = pp_config["agecols"]
            ages.index = self._obj.index
            for col in pp_config["agecols"]:
                self._obj[col] = ages[col]
                self._validate(self._obj)
            print("Ages added to data")
        else:
            print(
                f"Wrong shape {ages.shape} of copied data. Awaits {self._obj.shape} Set correct options and try again."
            )


if __name__ == "__main__":  # pragma: no cover
    pass
