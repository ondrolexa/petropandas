import importlib.resources
import json

import matplotlib.pyplot as plt
import mpltern  # NOQA
import numpy as np
import pandas as pd
import pyparsing
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pandas.api.types import is_numeric_dtype
from periodictable import elements
from periodictable.core import iselement, ision
from periodictable.formulas import formula

from petropandas.constants import (
    AGECOLS,
    COLNAMES,
    ISOPLOT,
    ISOPLOT_FORMATS,
    REE,
    REE_PLOT,
)
from petropandas.minerals import Mineral

config = {
    "isoplot_default_format": 2,
    "colnames": COLNAMES,
    "agecols": AGECOLS,
    "isoplot_formats": ISOPLOT_FORMATS,
    "ree_plot": REE_PLOT,
}

germ = importlib.resources.files("petropandas").joinpath("data", "germ.json")
with germ.open() as fp:
    config["reservoirs"] = json.load(fp)


def oxideprops(f):
    ncat, element = f.structure[0]
    noxy = f.atoms[elements.name("oxygen")]
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
            f"Column definition {tmpl} is not defined. Check `config['colnames']`"
        )


class NoEndMembers(Exception):
    def __init__(self, mineral):
        super().__init__(f"Mineral {mineral} has no endmembers method defined.")


@pd.api.extensions.register_dataframe_accessor("petro")
class PetroAccessor:
    """Use `.petro` pandas dataframe accessor."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def search(self, s, **kwargs) -> pd.DataFrame:
        """Select subset of data from dataframe containing string s in index or column.

        Note: Works only with non-numeric index or column

        Args:
            s (str): Returns all rows which contain string s in index or column.

        Keyword Args:
            on (str): Name of column used for search. When `None` the index is used
            regex (bool): If True, assumes the pat is a regular expression. If False,
                treats the pat as a literal string.

        Returns:
            Dataframe with selected data
        """
        on = kwargs.get("on", None)
        regex = kwargs.get("regex", True)
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

        Check `config['colnames']` for available templates. User-defined templates
        could be added. Template is a dict used for `pandas.DataFrame.rename`.

        Args:
            template (str): Name of renaming template

        Returns:
            Dataframe with renamed columns
        """
        if template not in config["colnames"]:
            raise TemplateNotDefined(template)
        return self._obj.rename(columns=config["colnames"][template])

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
        self._obj[expr if name is None else name] = self._obj.eval(expr)
        return self._obj

    def to_latex(self, total=False, transpose=True, precision=2) -> str:
        """Convert datatable to LaTeX string

        Args:
            total (bool, optional): Add column `"Total"` with total sums.
                Default `True`
            transpose (bool, optional): Place samples as columns. Default ``True``
            precision (bool, optional): Nimber of decimal places. Default 2

        """
        df = self._obj.copy()
        if total:
            df["Total"] = df.sum(axis=1, numeric_only=True)
        if transpose:
            df = df.T
        return df.fillna("").style.format(precision=precision).to_latex()


@pd.api.extensions.register_dataframe_accessor("petroplots")
class PetroPlotsAccessor:
    """Use `.petroplots` pandas dataframe accessor."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def garnet_profile(self, **kwargs):
        """Plot garnet profiles.

        Note:
            DataFrame must contains garnet endmembers properly sorted.

        Keyword Args:
            use_index (bool): When True, xticks are derived from DataFrame
                index, otherwise ticks are sequential. Default False
            xticks_rotation (float): Rotation of xtick labels. Default 0
            twin (bool): When ``True``, the plot has two independent y-axes
                for better scaling. Endmembers must be separated into two groups using
                `cols` and `cols_extra` args. When ``False`` both groups are plotted on
                same axes. Default ``True``
            cols (list): list of endmember names in first group. Default
                `["Alm"]`
            cols_extra (list): list of endmember names in second group.
                Default `["Prp", "Grs", "Sps"]`
            lim (tuple): y-axis limits for first axis or auto when
                ``None``. Default ``None``
            lim_extra (tuple): y-axis limits for second axis or auto when
                ``None``. Default ``None``
            percents (bool): When ``True`` y-axes scale is percents, otherwise fraction
            xlabel (str): label of the x-axis. Default auto.
            filename (str): When not ``None``, the plot is saved to file,
                otherwise the plot is shown.
            maxticks (int): maximum number of ticks on x-axis. Default 20
            xticks_rotation (int): rotation of xticks labels. Default 0
            markers (bool): Show markers. Default False
            subplot_kws (dict): kwargs passed to matplotlib subplots. Default {}

        """
        cols = kwargs.get("cols", ["Alm"])
        cols_extra = kwargs.get("cols_extra", ["Prp", "Sps", "Grs"])
        twin = kwargs.get("twin", True)
        margin = kwargs.get("margin", 0.1)
        lim = kwargs.get("lim", None)
        lim_extra = kwargs.get("lim_extra", None)
        filename = kwargs.get("filename", None)
        maxticks = kwargs.get("maxticks", 20)
        percents = kwargs.get("percents", False)
        use_index = kwargs.get("use_index", False)
        xlabel = kwargs.get("xlabel", None)
        xticks_rotation = kwargs.get("xticks_rotation", 0)
        markers = kwargs.get("markers", None)

        em = self._obj.copy()
        colors = sns.color_palette(None, len(cols) + len(cols_extra))
        fig, ax1 = plt.subplots(**kwargs.get("subplot_kws", {}))
        if percents:
            multiple = 100
            unit = " [%]"
        else:
            multiple = 1
            unit = " [fraction]"
        if use_index:
            xlabel = "index" if xlabel is None else xlabel
        else:
            em.index = range(len(em))
            xlabel = "position" if xlabel is None else xlabel
        ax1.set_xlabel(xlabel)
        if markers is None:
            markers1 = markers2 = None
        else:
            if len(markers) == 1:
                markers = (len(cols) + len(cols_extra)) * markers
            markers1 = markers[: len(cols)]
            markers2 = markers[len(cols) :]
        if twin:
            ax1.set_ylabel(" ".join(cols) + unit)
            with sns.color_palette(colors[: len(cols)]):
                lns1 = sns.lineplot(
                    multiple * em[cols], ax=ax1, markers=markers1, dashes=False
                )
            # h1 = ax1.plot(xvals, multiple * em[cols], marker=marker, ms=ms)
            if lim is not None:
                ax1.set_ylim(lim[0], lim[1])
            else:
                ax1.set_ymargin(margin)
            ax2 = ax1.twinx()
            ax2.set_ylabel(" ".join(cols_extra) + unit)
            with sns.color_palette(colors[len(cols) :]):
                lns2 = sns.lineplot(
                    multiple * em[cols_extra],
                    ax=ax2,
                    markers=markers2,
                    dashes=False,
                )
            # h2 = ax2.plot(xvals, multiple * em[cols_extra], marker=marker, ms=ms)
            if lim_extra is not None:
                ax2.set_ylim(lim_extra[0], lim_extra[1])
            else:
                ax2.set_ymargin(margin)
            # common legend
            h1, l1 = lns1.get_legend_handles_labels()
            h2, l2 = lns2.get_legend_handles_labels()
            ax1.legend(
                handles=h1 + h2,
                labels=l1 + l2,
                loc=8,
                bbox_to_anchor=(0.5, 1),
                ncol=4,
                title=None,
                frameon=False,
            )
            ax2.get_legend().remove()
        else:
            ax1.set_ylabel(unit[2:-1])
            with sns.color_palette(colors):
                lns1 = sns.lineplot(
                    multiple * em[cols + cols_extra],
                    ax=ax1,
                    markers=markers,
                    dashes=False,
                )
            # h1 = ax1.plot(xvals, multiple * em[cols + cols_extra], marker=marker, ms=ms)
            if lim is not None:
                ax1.set_ylim(lim[0], lim[1])
            else:
                ax1.set_ymargin(margin)
            sns.move_legend(
                ax1, loc=8, bbox_to_anchor=(0.5, 1), ncol=4, title=None, frameon=False
            )
        # Find at most maxticks ticks on the x-axis at 'nice' locations
        xloc = MaxNLocator(maxticks - 1, integer=True)
        ax1.xaxis.set_major_locator(xloc)
        ax1.tick_params(axis="x", labelrotation=xticks_rotation)
        fig.tight_layout()
        if filename is not None:
            fig.savefig(filename)
            print(f"{filename} saved.")
            plt.close(fig)
        else:
            plt.show()

    def ternary(self, top, left, right, **kwargs):
        """Ternary scatter plot

        Args:
            top (str): Name of column for top variable
            left (str): Name of column for left variable
            right (str): Name of column for right variable

        Keyword Args:
            kind (str): Kind of plot. One of `scatter`, `plot`
                Deafult `scatter`
            ternary_sum (float): Total sum. Default 1.0
            ax (Axes): matplotlib axes to be used.
            return_ax (bool): Whether to return matplotlib axes.
                Default False
            show (bool): Whether to show plot. Default True

        Additional keyword arguments are passed to underlying matplotlib
        function, e.g. `s` or `c` to `scatter` or `label`

        """
        kind = kwargs.pop("kind", "scatter")
        ternary_sum = kwargs.pop("ternary_sum", 1.0)
        return_ax = kwargs.pop("return_ax", False)
        show = kwargs.pop("show", True)
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="ternary", ternary_sum=ternary_sum)
        ax.set_tlabel(top)
        ax.set_llabel(left)
        ax.set_rlabel(right)
        top_vals = self._obj[top]
        left_vals = self._obj[left]
        right_vals = self._obj[right]
        leg_color = False
        leg_size = False
        tit_color = None
        tit_size = None
        match kind:
            case "scatter":
                if "s" in kwargs:
                    if isinstance(kwargs["s"], str):
                        tit_size = kwargs["s"]
                        kwargs["s"] = self._obj[kwargs["s"]]
                if "c" in kwargs:
                    if isinstance(kwargs["c"], str):
                        tit_color = kwargs["c"]
                        kwargs["c"] = self._obj[kwargs["c"]]
                pc = ax.scatter(top_vals, left_vals, right_vals, **kwargs)
                if "c" in kwargs:
                    handles, labels = pc.legend_elements(prop="colors", num=6)
                    if len(handles) > 1:
                        leg_color = ax.legend(
                            handles, labels, loc="upper right", title=tit_color
                        )
                if "s" in kwargs:
                    handles, labels = pc.legend_elements(prop="sizes", num=6)
                    if len(handles) > 1:
                        leg_size = ax.legend(
                            handles, labels, loc="upper left", title=tit_size
                        )
            case "plot":
                pc = ax.plot(top_vals, left_vals, right_vals, **kwargs)

        if return_ax:
            return ax
        if show:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
            else:
                if leg_color:
                    ax.add_artist(leg_color)
                if leg_size:
                    ax.add_artist(leg_size)
            plt.show()


@pd.api.extensions.register_dataframe_accessor("isoplot")
class IsoplotAccessor:
    """Use `.isoplot` pandas dataframe accessor."""

    def __init__(self, pandas_obj):
        self._names = []
        self._others = []
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        # verify there is a isoplot column
        valid = []
        for col in obj.columns:
            if col in ISOPLOT:
                valid.append(True)
                self._names.append(col)
            else:
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("isoplot")

    def df(self, **kwargs) -> pd.DataFrame:
        """Returns dataframe.

        Keyword Args:
            iso (int): IsoplotR format. Default `config["isoplot_default_format"]`
            C (str): Column to be used as color. Default None
            omit (str): Column to be used as omit. Default None
            comment (str): Column to be used as comment. Default None

        Returns:
            Dataframe
        """
        iso = kwargs.get("iso", config["isoplot_default_format"])
        df = self._obj[config["isoplot_formats"][iso]]
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
        return df

    def clipboard(self, **kwargs):
        """Copy data to clipbord to be used in IsoplotR online.

        Note:
            [IsoplotR online](http://isoplotr.es.ucl.ac.uk/home/index.html)

                Vermeesch, P., 2018, IsoplotR: a free and open toolbox for geochronology.
                Geoscience Frontiers, v.9, p.1479-1493, doi: 10.1016/j.gsf.2018.04.001.

        Keyword Args:
            iso (int): IsoplotR format. Default `config["isoplot_default_format"]`
            C (str): Column to be used as color. Default None
            omit (str): Column to be used as omit. Default None
            comment (str): Column to be used as comment. Default None
        """
        df = self.df(**kwargs)
        df.to_clipboard(header=False, index=False)

    def calc_ages(self, **kwargs) -> pd.DataFrame | None:
        """Copy data to clipbord, calc ages in IsoplotR online and paste back results.

        Keyword Args:
            iso (int): IsoplotR format. Default `config["isoplot_default_format"]`
            C (str): Column to be used as color. Default None
            omit (str): Column to be used as omit. Default None
            comment (str): Column to be used as comment. Default None

        Returns:
            Dataframe with calculated ages
        """
        iso = kwargs.get("iso", config["isoplot_default_format"])
        self.clipboard(**kwargs)
        print(f"Data in format {iso} copied to clipboard")
        print("Calc ages with Stacey-Kramers, discordance and digits 5")
        input("Then copy to clipboard and press Enter to continue...")
        ages = pd.read_clipboard(header=None)
        if ages.shape[1] == 9:
            ages.columns = config["agecols"]
            ages.index = self._obj.index
            for col in config["agecols"]:
                self._obj[col] = ages[col]
                self._validate(self._obj)
            print("Ages added to data")
        else:
            print(
                f"Wrong shape {ages.shape} of copied data. Awaits {self._obj.shape} Set correct options and try again."
            )


class AccessorTemplate:
    """Base class"""

    def __init__(self, pandas_obj):
        self._names = []
        self._others = []
        self._names_props = []
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @property
    def _df(self) -> pd.DataFrame:
        """Returns dataframe with appropriate columns."""
        return self._obj[self._names]

    def _validate(self, obj):
        raise NotImplementedError

    def _final(self, df, **kwargs) -> pd.DataFrame:
        # select = kwargs.get("select", [])
        # if select:
        #     df = df[df.columns.intersection(select)]
        #     rest = df.columns.symmetric_difference(select).difference(df.columns)
        #     df[rest] = np.nan
        keep = kwargs.get("keep", [])
        match keep:
            case []:
                return df
            case "all":
                return pd.concat([df, self._obj[self._others]], axis=1)
            case _:
                return pd.concat([df, self._obj[keep]], axis=1)

    @property
    def props(self) -> pd.DataFrame:
        """Returns properties."""
        return pd.DataFrame(self._names_props, index=pd.Index(self._names))

    def df(self, **kwargs) -> pd.DataFrame:
        """Returns dataframe.

        Keyword Args:
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe
        """
        return self._final(self._df, **kwargs)


@pd.api.extensions.register_dataframe_accessor("oxides")
class OxidesAccessor(AccessorTemplate):
    """Use `.oxides` pandas dataframe accessor."""

    def _validate(self, obj):
        # verify there is a oxides column
        valid = []
        for col in obj.columns:
            try:
                f = formula(col)
                if (len(f.atoms) == 2) and (elements.name("oxygen") in f.atoms):
                    valid.append(True)
                    self._names.append(col)
                    self._names_props.append(oxideprops(f))
                else:
                    valid.append(False)
                    self._others.append(col)
            except (ValueError, pyparsing.exceptions.ParseException):
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("oxides")

    def mean(self, **kwargs) -> pd.DataFrame:
        """Return Dataframe with single row of arithmetic means of oxide columns"""
        return pd.DataFrame(self._df.mean(axis=0)).T

    def scale(self, **kwargs) -> pd.DataFrame:
        """Normalize oxide values to given sum.

        Keyword Args:
            to (float): Sum of oxides. Default 100.0
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
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with molar proportions

        """
        res = self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def cat_number(self, **kwargs) -> pd.DataFrame:
        """Calculate cations number.

        Keyword Args:
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with molar proportions

        """
        res = self.props["ncat"] * self._df.div(self.props["mass"])
        return self._final(res, **kwargs)

    def oxy_number(self, **kwargs) -> pd.DataFrame:
        """Calculate oxygens number.

        Keyword Args:
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
            noxy (int): ideal number of oxygens. Default 1
            ncat (int): ideal number of cations. Default 1
            tocat (bool): when ``True`` normalized to ``ncat``,
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
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Apatite corrected dataframe

        """
        if ("P2O5" in self._names) and ("CaO" in self._names):
            df = self._df.div(self.props["mass"])
            df = df.div(df.sum(axis=1), axis=0)
            df["CaO"] = (df["CaO"] - (10 / 3) * df["P2O5"]).clip(lower=0)
            df = df.mul(self.props["mass"], axis=1)
            df = df.drop(columns="P2O5")
            df = df.div(df.sum(axis=1), axis=0).mul(self._df.sum(axis=1), axis=0)
            return self._final(df, **kwargs)
        else:
            return self._final(self._df, **kwargs)

    def convert_Fe(self, **kwargs) -> pd.DataFrame:
        """Recalculate FeO to Fe2O3 or vice-versa.

        Note:
            When only FeO exists, all is recalculated to Fe2O3. When only Fe2O3
            exists, all is recalculated to FeO. When both exists, Fe2O3 is
            recalculated and added to FeO. Otherwise datatable is not changed.

        Keyword Args:
            to (str): to what iron oxide Fe should be converted. Default `"FeO"`
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with converted Fe oxide

        """
        to = kwargs.get("to", "FeO")
        if to == "FeO":
            if "Fe2O3" in self._names:
                Fe3to2 = 2 * formula("FeO").mass / formula("Fe2O3").mass
                res = self._df.copy()
                if "FeO" in self._names:
                    res["FeO"] += Fe3to2 * res["Fe2O3"]
                else:
                    res["FeO"] = Fe3to2 * res["Fe2O3"]
                res = res.drop(columns="Fe2O3")
                return self._final(res, **kwargs)
            else:
                return self._final(self._df, **kwargs)
        elif to == "Fe2O3":
            if "FeO" in self._names:
                Fe2to3 = formula("Fe2O3").mass / formula("FeO").mass / 2
                res = self._df.copy()
                if "Fe2O3" in self._names:
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
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with recalculated Fe

        """
        charge = self.cat_number().mul(self.cnf(ncat), axis=0)
        if ("Fe2O3" in self._names) & ("FeO" not in self._names):
            charge["Fe2O3"].loc[pd.isna(self._df["Fe2O3"])] = 0
            chargedef = 2 * noxy - self.charges(ncat).sum(axis=1)
            toconv = chargedef
            charge["Fe2O3"] += toconv
            charge["FeO"] = -toconv
            ncats = self.props["ncat"]
            ncats["FeO"] = 1
            mws = self.props["mass"]
            mws["FeO"] = formula("FeO").mass
        elif "Fe2O3" in self._names:
            charge["Fe2O3"].loc[pd.isna(self._df["Fe2O3"])] = 0
            chargedef = 2 * noxy - self.charges(ncat).sum(axis=1)
            toconv = chargedef.clip(lower=0, upper=charge["FeO"])
            charge["Fe2O3"] += toconv
            charge["FeO"] = charge["FeO"] - toconv
            ncats = self.props["ncat"]
            mws = self.props["mass"]
        elif "FeO" in self._names:
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

    def apfu(self, mineral: Mineral, **kwargs) -> pd.DataFrame:
        """Calculate a.p.f.u for given mineral

        Args:
            mineral: Mineral instance (see `petropandas.minerals`)

        Keyword Args:
            force (bool): when True, remaining cations are added to last site
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
                res.append(mineral.apfu(row, force=force))
            return self._final(pd.DataFrame(res, index=self._obj.index), **kwargs)
        else:
            raise NoEndMembers(mineral)

    def check_stechiometry(self, mineral: Mineral, **kwargs) -> pd.Series:
        """Calculate average missfit of populated and ideal cations on sites for given mineral

        Args:
            mineral: Mineral instance (see `petropandas.minerals`)

        Keyword Args:
            force (bool): when True, remaining cations are added to last site
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
                res.append(mineral.check_stechiometry(row, force=force))
            return pd.Series(res, index=self._obj.index, name="misfit")
        else:
            raise NoEndMembers(mineral)

    def endmembers(self, mineral: Mineral, **kwargs) -> pd.DataFrame:
        """Calculate endmembers proportions

        Args:
            mineral: Mineral instance (see `petropandas.minerals`)

        Keyword Args:
            force (bool): when True, remaining cations are added to last site
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

        Keyword Args:
            H2O (float): wt% of water. When -1 the amount is calculated
                as 100 - Total. Default -1.
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

        Keyword Args:
            H2O (float): wt% of water. When -1 the amount is calculated
                as 100 - Total. Default -1.
            oxygen (float): value to calculate moles of ferric iron.
                Moles FeO = FeOtot - 2O and moles Fe2O3 = O. Default 0.01
            system (str): system to be used. One of 'MnNCKFMASHTO', 'NCKFMASHTO',
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

        Keyword Args:
            H2O (float): wt% of water. When -1 the amount is calculated
                as 100 - Total. Default -1.
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
class IonsAccessor(AccessorTemplate):
    """Use `.ions` pandas dataframe accessor."""

    def _validate(self, obj):
        # verify there is a ions column
        valid = []
        for col in obj.columns:
            try:
                f = formula(col)
                if (len(f.atoms) == 1) and (is_numeric_dtype(obj[col].dtype)):
                    if ision(next(iter(f.atoms.keys()))):
                        valid.append(True)
                        self._names.append(col)
                        self._names_props.append(elementprops(f))
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


@pd.api.extensions.register_dataframe_accessor("elements")
class ElementsAccessor(AccessorTemplate):
    """Use `.elements` pandas dataframe accessor."""

    def _validate(self, obj):
        # verify there is an element column
        valid = []
        for col in obj.columns:
            try:
                f = formula(col)
                if (len(f.atoms) == 1) and (is_numeric_dtype(obj[col].dtype)):
                    if iselement(next(iter(f.atoms.keys()))):
                        valid.append(True)
                        self._names.append(col)
                        self._names_props.append(elementprops(f))
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


@pd.api.extensions.register_dataframe_accessor("ree")
class REEAccessor(AccessorTemplate):
    """Use `.ree` pandas dataframe accessor."""

    def _validate(self, obj):
        # verify there is a REE column
        valid = []
        for col in obj.columns:
            if col in REE:
                valid.append(True)
                self._names.append(col)
                self._names_props.append(elementprops(formula(col)))
            else:
                valid.append(False)
                self._others.append(col)
        if not any(valid):
            raise MissingColumns("REE")

    def normalize(self, **kwargs) -> pd.DataFrame:
        """Normalize REE by reservoir.

        Note:
            Predefined reservoirs are imported from
            [GERM Reservoir Database](https://earthref.org/GERMRD/reservoirs/). You can
            check all available reservoirs in `config["reservoirs"]`.

        Keyword Args:
            reservoir (str): Name of reservoir. Deafult "CI Chondrites"
            reference (str): Reference. Default "McDonough & Sun 1995"
            source (str): Original source. Deafult same as reference.
            keep (list): list of additional columns to be included. Default [].

        Returns:
            Dataframe with normalized REE composition
        """
        reservoir = kwargs.get("reservoir", "CI Chondrites")
        reference = kwargs.get("reference", "McDonough & Sun 1995")
        source = kwargs.get("source", reference)
        nrm = pd.Series(config["reservoirs"][reservoir][reference][source])
        res = self._df / nrm
        res = res[self._names]
        res["Eu/Eu*"] = res["Eu"] / np.sqrt(res["Sm"] * res["Gd"])
        res["Gd/Yb"] = res["Gd"] / res["Yb"]
        return self._final(res, **kwargs)

    def plot(self, **kwargs):
        """Spiderplot of REE data.

        Note:
            List of REE used for plot could be set in `config["ree_plot"]`

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
            filename (str): If not none, plot is saved to file. Default None.
            dpi (int): DPI used for `savefig`. Default 150.
            keep (list): list of additional columns to be included. Default [].
        """
        fig, ax = plt.subplots()
        ndf = self.normalize(**kwargs)
        ax.set(yscale="log")
        ree = ndf.melt(
            id_vars=kwargs.get("keep", []),
            var_name="ree",
            ignore_index=False,
        )
        # select only REE for plotting
        ree = ree.loc[ree["ree"].isin(config["ree_plot"])]
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


if __name__ == "__main__":  # pragma: no cover
    pass
