"""DataFrame accessors for petrological microprobe data."""

from __future__ import annotations

import warnings

import pandas as pd

import petropandas._calc as _calc
from petropandas._config import ppconfig
from petropandas._core import ALIASES
from petropandas._core import _formula_cols as _core_formula_cols
from petropandas._core import _is_oxide as _core_is_oxide
from petropandas._core import _oxide_cols as _core_oxide_cols
from petropandas._minerals import Mineral


# ---------------------------------------------------------------------------
# Canonical column order
# ---------------------------------------------------------------------------

MAJOR_OXIDES: list[str] = [
    "SiO2",
    "TiO2",
    "Al2O3",
    "FeO",
    "Fe2O3",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
]

VOLATILES: list[str] = [
    "H2O",
    "CO2",
    "SO3",
]

# ---------------------------------------------------------------------------
# Thermodynamic software bulk-composition column orders
# ---------------------------------------------------------------------------

# fmt: off
_TC_SYSTEMS: dict[str, list[str]] = {
    "MnNCKFMASHTO": ["H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "MnO", "O"],
    "NCKFMASHTO":   ["H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O"],
    "KFMASH":       ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"],
    "NCKFMASHTOCr": ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3"],
    "NCKFMASTOCr":  ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "TiO2", "O", "Cr2O3"],
}

_PERPLEX_SYSTEMS: dict[str, list[str]] = {
    "MnNCKFMASHTO": ["H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "MnO", "O2"],
    "NCKFMASHTO":   ["H2O", "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O2"],
    "KFMASH":       ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"],
    "NCKFMASHTOCr": ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O2", "Cr2O3"],
    "NCKFMASTOCr":  ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "TiO2", "O2", "Cr2O3"],
}

_MAGEMIN_SYSTEMS: dict[str, list[str]] = {
    "ig":  ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "H2O"],
    "mp":  ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "MnO", "H2O"],
    "mb":  ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "H2O"],
    "um":  ["SiO2", "Al2O3", "MgO", "FeO", "O", "H2O", "S"],
    "ume": ["SiO2", "Al2O3", "MgO", "FeO", "O", "H2O", "S", "CaO", "Na2O"],
    "mpe": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "MnO", "H2O", "CO2", "S"],
    "mbe": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "H2O"],
    "mtl": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Na2O"],
}
# fmt: on


def _sort_oxide_columns(columns: list[str]) -> list[str]:
    """Re-order oxide columns in standard petrological order.

    Major oxides come first in a fixed order, then remaining oxides
    alphabetically, then volatiles, then any non-oxide columns.
    """
    col_set = set(columns)
    major = [c for c in MAJOR_OXIDES if c in col_set]
    volatile = [c for c in VOLATILES if c in col_set]
    oxides = {c for c in columns if _core_is_oxide(c)}
    other_oxides = sorted(oxides - set(major) - set(volatile))
    non_oxides = sorted(col_set - oxides)
    return major + other_oxides + volatile + non_oxides


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _needs_cleanup(obj: pd.DataFrame) -> bool:
    """True when DataFrame is raw user data (no petro_units set)."""
    return "petro_units" not in obj.attrs


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, apply ALIASES, fill NaN, clip negatives on formula columns."""
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    work = work.rename(columns=ALIASES)
    formula_cols = _core_formula_cols(work)
    if formula_cols:
        work[formula_cols] = work[formula_cols].fillna(0).clip(lower=0)
    return work


def _reframe_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return *df* restricted to *columns*, filling any missing ones with 0.0."""
    return pd.DataFrame(
        {col: df[col] if col in df.columns else 0.0 for col in columns},
        index=df.index,
    )


class _CleaningAccessor:
    """Shared constructor for all petropandas DataFrame accessors.

    Cleans raw user data (strip column names, apply ``ALIASES``, fill/clip
    formula columns) on first access; DataFrames that already carry
    ``petro_units`` are assumed already clean and are just copied.
    """

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = _clean_df(obj) if _needs_cleanup(obj) else obj.copy()


# ---------------------------------------------------------------------------
# MineralAccessor — general tools + mineral calculations
# ---------------------------------------------------------------------------


@pd.api.extensions.register_dataframe_accessor("mineral")
class MineralAccessor(_CleaningAccessor):
    """General-purpose accessor for microprobe DataFrames (``df.mineral``)."""

    # -- helpers -------------------------------------------------------------

    def _oxide_cols(self) -> list[str]:
        return _core_oxide_cols(self._obj)

    def _units(self) -> str:
        return self._obj.attrs.get("petro_units", "wt%")

    # -- mineral calculations ------------------------------------------------

    def apfu(self, mineral: Mineral) -> pd.DataFrame:
        """Compute element APFU for a mineral, with valence splits."""
        return mineral.apfu(self._obj, units=self._units())

    def site_allocations(self, mineral: Mineral) -> pd.DataFrame:
        """Compute site allocations with hierarchical (site, cation) columns."""
        return mineral.site_allocations(self._obj, units=self._units())

    def end_members(self, mineral: Mineral, **kwargs: object) -> pd.DataFrame:
        """Compute end-member proportions for a mineral.

        Extra keyword arguments (e.g. ``order_parameters`` for hpxeos
        ``Phase`` order-disorder variables) are forwarded to the mineral's
        own ``end_members`` method.
        """
        return mineral.end_members(self._obj, units=self._units(), **kwargs)

    def check_stoichiometry(self, mineral: Mineral) -> pd.DataFrame:
        """Validate analysis against a mineral's ideal stoichiometry.

        Args:
            mineral: Mineral instance providing ideal ranges, sites, and
                valence-split configuration.

        Returns:
            DataFrame with the same index as the input, one column per
            criterion, values between 0 (impossible) and 1 (perfect fit).

        Columns in the returned DataFrame:

        * ``analytical_total`` — Oxide wt% sum vs mineral-specific ideal range.
        * ``cation_deviation`` — Total APFU vs ideal cation count (NaN if
          mineral does not define ``ideal_cations``).
        * ``charge_balance`` — Total positive charge vs expected from oxygen
          count (exponential decay).
        * ``fe3+_validity`` — Binary check that ``Fe{3+}`` and ``Fe{2+}`` are
          non-negative after valence splitting (NaN if no Fe split).
        * ``site_vacancies`` — Mean site occupancy fraction across all sites.
        * ``leftover_cations`` — Fraction of total APFU not assigned to any
          site.
        * ``tetrahedral_fill`` — T-site sum vs T-site capacity (NaN if no
          T-site defined).
        """
        df = self._obj
        oxide_cols = self._oxide_cols()
        units = self._units()

        try:
            apfu_df = mineral._raw_apfu(df, units=units)
            fe_split_ok = True
        except KeyError:
            apfu_df = _calc.to_apfu(df, n_oxygens=mineral.n_oxygens, units=units)
            fe_split_ok = False
        site_alloc = mineral._allocate_sites(apfu_df)

        result = pd.DataFrame(
            {
                "analytical_total": _calc.score_analytical_total(
                    df[oxide_cols].sum(axis=1), mineral.analytical_total_range
                ),
                "cation_deviation": _calc.score_cation_deviation(
                    apfu_df, mineral.ideal_cations
                ),
                "charge_balance": _calc.score_charge_balance(
                    apfu_df, mineral.n_oxygens
                ),
                "fe3+_validity": _calc.score_fe3_validity(apfu_df, fe_split_ok),
                "site_vacancies": _calc.score_site_vacancies(
                    site_alloc, mineral.site_definitions
                ),
                "leftover_cations": _calc.score_leftover_cations(apfu_df, site_alloc),
                "tetrahedral_fill": _calc.score_tetrahedral_fill(
                    site_alloc, mineral.site_definitions
                ),
            },
            index=df.index,
        )
        return result.dropna(axis=1, how="all")


# ---------------------------------------------------------------------------
# OxidesAccessor — callable, returns oxide wt% DataFrame
# ---------------------------------------------------------------------------


@pd.api.extensions.register_dataframe_accessor("oxides")
class OxidesAccessor(_CleaningAccessor):
    """Callable accessor (``df.oxides``).

    ``df.oxides()`` returns a copy of the DataFrame containing only
    recognised oxide columns, converted to wt% if necessary.
    """

    def __call__(self) -> pd.DataFrame:
        """Return a copy with only recognised oxide columns in wt%."""
        units = self._obj.attrs.get("petro_units", "wt%")
        result = _calc.convert(self._obj, "wt%", from_unit=units)
        cols = _core_oxide_cols(result)
        out = result[cols].copy()
        out.attrs["petro_units"] = "wt%"
        return out

    def sorted(self) -> pd.DataFrame:
        """Return oxide wt% with columns in standard petrological order.

        Major oxides first (SiO2, TiO2, Al2O3, FeO, Fe2O3, MnO,
        MgO, CaO, Na2O, K2O, P2O5), then other oxides alphabetically,
        then volatiles (H2O, CO2, SO3).
        """
        out = self()
        out = out[_sort_oxide_columns(list(out.columns))]
        return out

    def normalized(self) -> pd.DataFrame:
        """Normalise oxide wt% so each row sums to 100 %."""
        result = _calc.normalize(self())
        result.attrs["petro_units"] = "wt%"
        return result

    def mean(self, *, groupby: str | None = None) -> pd.DataFrame:
        """Compute mean oxide wt% across rows.

        Args:
            groupby: Column name to group by.  When *None* (default)
                a single-row DataFrame with the overall mean is
                returned.  When a column name is given, one row per
                group is returned with the group label as index.

        Returns:
            DataFrame with mean oxide values.
        """
        out = self()
        if groupby is not None:
            if groupby not in self._obj.columns:
                msg = f"Groupby column {groupby!r} not found in DataFrame"
                raise ValueError(msg)
            result = self._obj.groupby(groupby)[list(out.columns)].mean()
        else:
            result = pd.DataFrame(
                {col: [out[col].mean()] for col in out.columns},
            )
        result.attrs["petro_units"] = "wt%"
        return result

    def split_valence(
        self,
        element: str,
        method: str,
        n_oxygens: int | float,
        ideal_cations: int | float,
    ) -> pd.DataFrame:
        """Split an element into low/high-charge oxide columns.

        Converts to APFU, applies the valence split, then converts
        back to oxide wt%.

        Args:
            element: Element symbol to split (e.g. ``"Fe"``).
            method: Splitting method (``"droop"`` or ``"schumacher"``).
            n_oxygens: Oxygen denominator for APFU normalisation.
            ideal_cations: Ideal cation total for APFU normalisation.

        Returns:
            DataFrame with split oxide columns (e.g. ``"FeO"``, ``"Fe2O3"``).
        """
        units = self._obj.attrs.get("petro_units", "wt%")
        apfu_df = _calc.convert(self._obj, "apfu", from_unit=units, n_oxygens=n_oxygens)
        result_apfu = _calc.split_valence(
            apfu_df,
            element=element,
            method=method,
            n_oxygens=n_oxygens,
            ideal_cations=ideal_cations,
        )
        total = self._obj.attrs.get("petro_total")
        if total is None:
            oxide_cols = _core_oxide_cols(self._obj)
            total = self._obj[oxide_cols].sum(axis=1) if oxide_cols else None
        result = _calc.from_apfu(result_apfu, n_oxygens=n_oxygens, total=total)
        cols = _core_oxide_cols(result)
        out = result[cols].copy()
        out.attrs["petro_units"] = "wt%"
        return out

    def oxidize(self, o_excess: float | pd.Series) -> pd.DataFrame:
        """Split total Fe into FeO and Fe2O3 based on excess oxygen.

        Converts to molar proportions, applies the THERMOCALC
        excess-oxygen split, then converts back to wt%.

        Args:
            o_excess: Mole percent of excess oxygen relative to total
                oxide moles.

        Returns:
            DataFrame with split oxide columns in wt%.
        """
        moles_df = _calc.convert(self(), "moles")
        result_moles = _calc.oxidize_moles(moles_df, o_excess)
        result = _calc.convert(result_moles, "wt%", from_unit="moles")
        result.attrs["petro_units"] = "wt%"
        return result

    def reduce(self) -> pd.DataFrame:
        """Merge Fe₂O₃ into FeO, converting to FeO equivalent.

        If no Fe₂O₃ column exists, returns a copy unchanged.

        Returns:
            DataFrame with all iron as FeO in wt%.
        """
        result = _calc.fe2o3_to_feo(self())
        result.attrs["petro_units"] = "wt%"
        return result

    def apatite_correction(self) -> pd.DataFrame:
        """Remove CaO bound in apatite and zero P₂O₅.

        Uses true fluorapatite stoichiometry Ca₅(PO₄)₃F (Ca:P = 5:3).
        Useful before thermodynamic modelling where P-bearing phases
        are absent or inaccurate.

        Returns:
            Corrected copy with reduced CaO and P₂O₅ = 0.
        """
        result = _calc.apatite_correction(self())
        result.attrs["petro_units"] = "wt%"
        return result

    def select(
        self,
        arg: str | list | pd.Series,
        *,
        on: str | None = None,
    ) -> pd.DataFrame:
        """Filter rows by index, column values, or boolean mask.

        Args:
            arg: Selection criterion:

                - ``str`` — keep rows where the target *contains* the
                  substring (case-sensitive).
                - ``list`` — keep rows where the target *is in* the list.
                - ``pd.Series`` (bool) — logical row indexing; the Series
                  must have the same index as the DataFrame.

            on: Column whose values are tested against *arg*.  When
                *None* (default), the DataFrame index is tested instead.
                Ignored when *arg* is a boolean Series.

        Returns:
            Filtered copy of the DataFrame with ``petro_units`` preserved.
        """
        if isinstance(arg, pd.Series):
            if not arg.index.equals(self._obj.index):
                msg = (
                    "Boolean Series index does not match DataFrame index; "
                    "reindex before calling select()"
                )
                raise ValueError(msg)
            out = self._obj.loc[arg]
        elif isinstance(arg, str):
            if on is None and self._obj.index.dtype.kind in ("i", "f"):
                warnings.warn(
                    "Index is numeric; select() with a string argument "
                    "may not match as expected",
                    UserWarning,
                    stacklevel=2,
                )
            target = self._obj.index if on is None else self._obj[on]
            mask = target.astype(str).str.contains(arg)
            out = self._obj.loc[mask]
        elif isinstance(arg, list):
            target = self._obj.index if on is None else self._obj[on]
            mask = target.isin(arg)
            out = self._obj.loc[mask]
        else:
            msg = f"arg must be str, list, or Series, got {type(arg).__name__}"
            raise TypeError(msg)
        out = out.copy()
        out.attrs["petro_units"] = self._obj.attrs.get("petro_units", "wt%")
        return out


# ---------------------------------------------------------------------------
# MolesAccessor — callable, returns molar-proportions DataFrame
# ---------------------------------------------------------------------------


@pd.api.extensions.register_dataframe_accessor("moles")
class MolesAccessor(_CleaningAccessor):
    """Callable accessor (``df.moles``).

    ``df.moles()`` returns a copy with oxide columns as molar proportions.
    Auto-converts from the current ``petro_units`` (default: wt%).
    """

    def __call__(self) -> pd.DataFrame:
        """Return oxide columns as molar proportions."""
        units = self._obj.attrs.get("petro_units", "wt%")
        result = _calc.convert(self._obj, "moles", from_unit=units)
        cols = _core_formula_cols(result)
        out = result[cols].copy()
        out.attrs["petro_units"] = "moles"
        return out

    def normalized(self) -> pd.DataFrame:
        """Normalize moles to 100 mol%."""
        result = _calc.normalize(self())
        result.attrs["petro_units"] = "moles"
        return result


# ---------------------------------------------------------------------------
# CationsAccessor — callable, returns APFU DataFrame
# ---------------------------------------------------------------------------


@pd.api.extensions.register_dataframe_accessor("cations")
class CationsAccessor(_CleaningAccessor):
    """Callable accessor (``df.cations``).

    ``df.cations(n_oxygens=N)`` or ``df.cations(n_cations=N)`` returns atoms
    per formula unit.  Auto-converts from the current ``petro_units``
    (default: wt%).
    """

    def __call__(
        self,
        *,
        n_oxygens: int | float | None = None,
        n_cations: int | float | None = None,
    ) -> pd.DataFrame:
        """Convert to atoms per formula unit.

        Args:
            n_oxygens: Fixed oxygen denominator (e.g. 6 for pyroxene).
                Mutually exclusive with *n_cations*.
            n_cations: Fixed cation total (e.g. 4 for feldspar).
                Mutually exclusive with *n_oxygens*.

        Returns:
            DataFrame with ion-named columns (e.g. ``"Si{4+}"``, ``"Fe{2+}"``).
        """
        units = self._obj.attrs.get("petro_units", "wt%")
        if units == "apfu":
            return self._obj
        oxide_cols = _core_oxide_cols(self._obj)
        total = self._obj[oxide_cols].sum(axis=1) if oxide_cols else None
        result = _calc.convert(
            self._obj,
            "apfu",
            from_unit=units,
            n_oxygens=n_oxygens,
            n_cations=n_cations,
        )
        result.attrs["petro_units"] = "apfu"
        result.attrs["petro_n_oxygens"] = n_oxygens
        result.attrs["petro_n_cations"] = n_cations
        result.attrs["petro_total"] = total
        return result


# ---------------------------------------------------------------------------
# BulkAccessor — bulk rock geochemistry
# ---------------------------------------------------------------------------


@pd.api.extensions.register_dataframe_accessor("bulk")
class BulkAccessor(_CleaningAccessor):
    """Bulk-rock geochemistry (``df.bulk``).

    ``df.bulk()`` returns a cleaned copy of the DataFrame in wt%.
    Provides normative mineralogy, alumina saturation, and oxide
    ratio calculations for whole-rock major-oxide data.
    """

    def __call__(self) -> pd.DataFrame:
        """Return a cleaned copy with only oxide and element columns in wt%."""
        cols = _core_formula_cols(self._obj)
        return self._obj[cols].copy()

    def normalized(self) -> pd.DataFrame:
        """Normalise wt% so each row sums to 100 %."""
        result = _calc.normalize(self())
        result.attrs["petro_units"] = "wt%"
        return result

    def reframe(self, columns: list[str]) -> pd.DataFrame:
        """Return a DataFrame with exactly the requested columns.

        Columns present in the underlying data are kept as-is.
        Missing columns are filled with zeros.

        Args:
            columns: Ordered list of column names for the output.

        Returns:
            DataFrame with the same index as the input and columns
            in the requested order.
        """
        return _reframe_columns(self._obj, columns)

    def cipw_simple(self) -> pd.DataFrame:
        """Compute CIPW normative mineralogy (simple version).

        Requires major oxides in wt% (at minimum SiO₂, Al₂O₃,
        Fe₂O₃, FeO, MgO, CaO, Na₂O, K₂O).  Missing optional
        oxides default to 0.

        Returns:
            DataFrame with normative mineral columns (Qz, Or, Ab,
            An, Di, Hy, Mt, Il, Ap, etc.) in wt%.
        """
        return _calc.cipw_norm_simple(self())

    def cipw(
        self,
        *,
        normsum: bool = False,
        cancrinite: bool = False,
        spinel: bool = False,
        complete_results: bool = False,
    ) -> pd.DataFrame:
        """Compute the standard CIPW norm (GCDkit-faithful).

        Port of CIPW() from GCDkit/inst/Norms/CIPW.r.  Requires major
        oxides in wt% (at minimum SiO₂, Al₂O₃, Fe₂O₃, FeO, MgO, CaO,
        Na₂O, K₂O).  Missing optional oxides default to 0.

        Args:
            normsum: If True, normalise the norm so the mineral sum
                equals 100.
            cancrinite: If True, form cancrinite (Nc) from Na₂O + CO₂
                before the calcite step.
            spinel: If True, form spinel (Sp) from corundum + (Mg,Fe)O
                when SiO₂ < 45 (molar).
            complete_results: If True, keep all normative-mineral columns
                including sub-mineral splits (En, Fs, Fo, Fa, MgDi,
                FeDi) and columns that are zero for every sample.

        Returns:
            DataFrame indexed like the input, with normative-mineral
            columns in wt% named per GCDkit convention (Q, Or, Ab, An,
            Di, Hy, Mt, Il, Ap, …) plus a ``Total`` column.
        """
        return _calc.cipw_norm(
            self(),
            normsum=normsum,
            cancrinite=cancrinite,
            spinel=spinel,
            complete_results=complete_results,
        )

    def cipwhb(
        self,
        *,
        normsum: bool = False,
        cancrinite: bool = False,
        spinel: bool = False,
        complete_results: bool = False,
    ) -> pd.DataFrame:
        """Compute the CIPW norm with hornblende/biotite recasting.

        Port of CIPWhb() from GCDkit/inst/Norms/CIPWhb.r.  Mafic
        components are recast into biotite (Bi) and hornblende (Hbl)
        instead of diopside/hypersthene/olivine.

        Args:
            normsum: If True, normalise the norm so the mineral sum
                equals 100.
            cancrinite: If True, form cancrinite (Nc) from Na₂O + CO₂.
            spinel: If True, form spinel when SiO₂ < 45 (molar).
            complete_results: If True, keep all normative-mineral columns
                including sub-mineral splits and zero-only columns.

        Returns:
            DataFrame indexed like the input, with normative-mineral
            columns in wt% named per GCDkit convention (Q, Or, Ab, An,
            Bi, Hbl, Act, Ed, …) plus a ``Total`` column.
        """
        return _calc.cipw_norm_hb(
            self(),
            normsum=normsum,
            cancrinite=cancrinite,
            spinel=spinel,
            complete_results=complete_results,
        )

    def alumina_saturation(self, classify: bool = False) -> pd.DataFrame:
        """Compute alumina saturation indices (A/NK, A/CNK).

        Molar ratios based on bulk oxide wt%.

        Args:
            classify: If True, add a ``"shand_class"`` column with
                ``"peraluminous"``, ``"metaluminous"``, or
                ``"peralkaline"``.

        Returns:
            DataFrame with ``A/NK`` and ``A/CNK`` columns.
        """
        result = _calc.alumina_saturation(self._obj)
        if classify:
            a_cnk = result["A/CNK"]
            a_nk = result["A/NK"]
            classes = pd.Series("metaluminous", index=result.index)
            classes[a_cnk >= 1.0] = "peraluminous"
            classes[a_nk < 1.0] = "peralkaline"
            result["shand_class"] = classes
        return result

    def oxide_ratios(self) -> pd.DataFrame:
        """Compute common bulk-rock oxide ratios.

        Returns a DataFrame with derived ratios including Mg#, FeOT,
        total alkalis, and K/Na.  Only ratios whose required oxides
        are all present are computed; missing ratios become NaN.

        Returns:
            DataFrame with ratio columns.
        """
        return _calc.oxide_ratios(self())

    def mean(
        self, *, groupby: str | None = None, weights: str | None = None
    ) -> pd.DataFrame:
        """Compute mean oxide wt% across rows.

        Args:
            groupby: Column name to group by. When *None* (default) a
                single-row DataFrame with the overall mean is returned.
                When given, one row per group is returned with the group
                label as index.
            weights: Column name to use as weights for a weighted mean.
                When *None* (default), an unweighted arithmetic mean is
                computed. The weights column is excluded from the result.

        Returns:
            DataFrame with mean oxide values.

        Raises:
            ValueError: If *groupby* or *weights* names a missing column.
        """
        if groupby is not None and groupby not in self._obj.columns:
            msg = f"Groupby column {groupby!r} not found in DataFrame"
            raise ValueError(msg)
        if weights is not None and weights not in self._obj.columns:
            msg = f"Weights column {weights!r} not found in DataFrame"
            raise ValueError(msg)

        cols = _core_formula_cols(self._obj)

        if weights is None:
            if groupby is not None:
                result = self._obj.groupby(groupby)[cols].mean()
            else:
                result = pd.DataFrame({col: [self._obj[col].mean()] for col in cols})
        else:
            weighted = self._obj[cols].mul(self._obj[weights], axis=0)
            if groupby is not None:
                weight_sums = self._obj.groupby(groupby)[weights].sum()
                grouped = weighted.groupby(self._obj[groupby]).sum()
                result = grouped.div(weight_sums, axis=0)
            else:
                weight_sum = self._obj[weights].sum()
                result = pd.DataFrame(
                    {col: [weighted[col].sum() / weight_sum] for col in cols}
                )

        result.attrs["petro_units"] = "wt%"
        return result

    # ------------------------------------------------------------------
    # Thermodynamic software bulk formatting
    # ------------------------------------------------------------------

    def _thermo_bulk_prep(
        self,
        system_cols: list[str],
        *,
        oxygen_key: str = "O",
        oxygen_mult: float = 1.0,
        use_molprop: bool = True,
        oxygen: float = 0.01,
        H2O: float = -1.0,
    ) -> pd.DataFrame:
        """Prepare bulk composition for thermodynamic software output.

        Reduces Fe₂O₃ → FeO, applies apatite correction, handles
        optional H₂O, converts to molar proportions (or wt%), and
        scales so the oxide total leaves room for the oxygen column.

        Args:
            system_cols: Ordered list of component names for the target
                software system.
            oxygen_key: Column label for the ferric-oxygen component
                (``"O"`` or ``"O2"``).
            oxygen_mult: Multiplier on the oxygen value (1 for O, 2 for O₂).
            use_molprop: When True convert to molar proportions before
                scaling; otherwise scale wt% directly.
            oxygen: Moles of ferric oxygen to add (default 0.01).
            H2O: Water content.  ``-1`` (default) computes the deficit
                as ``max(0, 100 − total)``.  A positive value is treated
                as the target H₂O wt% of the final bulk.

        Returns:
            DataFrame with columns in *system_cols* order.
        """
        work = _calc.fe2o3_to_feo(self())
        work = _calc.apatite_correction(work)

        # ---- H₂O handling --------------------------------------------------
        if "H2O" in system_cols and "H2O" not in work.columns:
            total = work.sum(axis=1)
            if H2O == -1:
                h2o = (100.0 - total).clip(lower=0.0)
            else:
                h2o = H2O * total / (100.0 - H2O)
            work["H2O"] = h2o

        # ---- select available columns --------------------------------------
        use = work.columns.intersection(system_cols)
        work = work[list(use)]

        # ---- convert to moles / scale to target ----------------------------
        target = 100.0 - oxygen * oxygen_mult
        if use_molprop:
            work = _calc.to_moles(work)
        work = work.div(work.sum(axis=1), axis=0) * target

        # ---- oxygen column -------------------------------------------------
        if oxygen_key in system_cols:
            work[oxygen_key] = oxygen * oxygen_mult

        return _reframe_columns(work, system_cols)

    # -- THERMOCALC ----------------------------------------------------------

    def TCbulk(
        self,
        *,
        system: str = ppconfig.default_system,
        oxygen: float = ppconfig.default_oxygen,
        H2O: float = ppconfig.default_H2O,
        dataframe: bool = False,
    ) -> pd.DataFrame | None:
        """Print oxides formatted as a THERMOCALC bulk script.

        CaO is corrected for apatite (P₂O₅-bound Ca removed, P₂O₅ zeroed).
        Fe₂O₃ is reduced to FeO before formatting.

        Args:
            system: Thermodynamic system.  One of ``"MnNCKFMASHTO"``,
                ``"NCKFMASHTO"``, ``"KFMASH"``, ``"NCKFMASHTOCr"``,
                ``"NCKFMASTOCr"``.  Default ``"MnNCKFMASHTO"``.
            oxygen: Moles of ferric oxygen (default 0.01).
            H2O: Water wt%.  ``-1`` (default) auto-calculates as the
                deficit from 100 wt%.
            dataframe: When True return a DataFrame instead of printing.

        Returns:
            DataFrame when *dataframe* is True, otherwise None (printed).
        """
        if system not in _TC_SYSTEMS:
            msg = f"Invalid system: {system!r} (choose from {sorted(_TC_SYSTEMS)})"
            raise ValueError(msg)

        cols = _TC_SYSTEMS[system]
        df = self._thermo_bulk_prep(
            cols,
            oxygen_key="O",
            oxygen_mult=1,
            use_molprop=True,
            oxygen=oxygen,
            H2O=H2O,
        )

        if dataframe:
            return df

        header = "bulk" + "".join(f"{lbl:>7}" for lbl in cols)
        print(header)
        for ix, row in df.iterrows():
            print("bulk" + "".join(f" {v:6.3f}" for v in row.values) + f"  % {ix}")
        return None

    # -- PerpleX -------------------------------------------------------------

    def Perplexbulk(
        self,
        *,
        system: str = ppconfig.default_system,
        oxygen: float = ppconfig.default_oxygen,
        H2O: float = ppconfig.default_H2O,
        dataframe: bool = False,
    ) -> pd.DataFrame | None:
        """Print oxides formatted as a PerpleX thermodynamic component list.

        CaO is corrected for apatite (P₂O₅-bound Ca removed, P₂O₅ zeroed).
        Fe₂O₃ is reduced to FeO before formatting.

        Args:
            system: Thermodynamic system.  One of ``"MnNCKFMASHTO"``,
                ``"NCKFMASHTO"``, ``"KFMASH"``, ``"NCKFMASHTOCr"``,
                ``"NCKFMASTOCr"``.  Default ``"MnNCKFMASHTO"``.
            oxygen: Moles of ferric O₂ (default 0.01, stored as O₂ = 0.02).
            H2O: Water wt%.  ``-1`` (default) auto-calculates as the
                deficit from 100 wt%.
            dataframe: When True return a DataFrame instead of printing.

        Returns:
            DataFrame when *dataframe* is True, otherwise None (printed).
        """
        if system not in _PERPLEX_SYSTEMS:
            msg = f"Invalid system: {system!r} (choose from {sorted(_PERPLEX_SYSTEMS)})"
            raise ValueError(msg)

        cols = _PERPLEX_SYSTEMS[system]
        df = self._thermo_bulk_prep(
            cols,
            oxygen_key="O2",
            oxygen_mult=2,
            use_molprop=True,
            oxygen=oxygen,
            H2O=H2O,
        )

        if dataframe:
            return df

        print("begin thermodynamic component list")
        for ox, val in df.iloc[0].items():
            print(f"{ox:6s}1 {val:8.5f}      0.00000      0.00000     molar amount")
        print("end thermodynamic component list")
        return None

    # -- MAGEMin -------------------------------------------------------------

    def MAGEMin(
        self,
        *,
        db: str = ppconfig.default_db,
        sys_in: str = ppconfig.default_sys_in,
        oxygen: float = ppconfig.default_oxygen,
        H2O: float = ppconfig.default_H2O,
        title: str | None = None,
        comment: str = "petropandas",
        dataframe: bool = False,
    ) -> pd.DataFrame | None:
        """Print oxides formatted as a MAGEMin bulk input file.

        CaO is corrected for apatite (P₂O₅-bound Ca removed, P₂O₅ zeroed).
        Fe₂O₃ is reduced to FeO before formatting.

        Args:
            db: MAGEMin database — ``"ig"`` (igneous), ``"mp"``
                (metapelite), ``"mb"`` (metabasite), ``"um"`` (ultramafic),
                ``"ume"`` (ultramafic ext.), ``"mpe"`` (metapelite ext.),
                ``"mbe"`` (metabasite ext.), ``"mtl"`` (mantle).
                Default ``"mp"``.
            sys_in: Unit system — ``"mol"`` (molar proportions, default)
                or ``"wt"`` (weight percent).
            oxygen: Moles of ferric oxygen (default 0.01).
            H2O: Water wt%.  ``-1`` (default) auto-calculates as the
                deficit from 100 wt%.
            title: Title string.  Defaults to the DataFrame index label.
            comment: Comment string (default ``"petropandas"``).
            dataframe: When True return a DataFrame instead of printing.

        Returns:
            DataFrame when *dataframe* is True, otherwise None (printed).
        """
        if db not in _MAGEMIN_SYSTEMS:
            msg = f"Invalid database: {db!r} (choose from {sorted(_MAGEMIN_SYSTEMS)})"
            raise ValueError(msg)

        cols = _MAGEMIN_SYSTEMS[db]
        df = self._thermo_bulk_prep(
            cols,
            oxygen_key="O",
            oxygen_mult=1,
            use_molprop=(sys_in == "mol"),
            oxygen=oxygen,
            H2O=H2O,
        )

        if dataframe:
            return df

        print("# HEADER")
        print("title; comments; db; sysUnit; oxide; frac; frac2")
        print("# BULK-ROCK COMPOSITION")
        for ix, row in df.iterrows():
            oxides = ", ".join(row.keys())
            values = ", ".join(f"{v:.4f}" for v in row.values)
            lbl = title if title is not None else ix
            print(f"{lbl};{comment};{db};{sys_in};[{oxides}];[{values}];")
        return None
