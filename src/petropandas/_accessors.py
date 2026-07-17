"""DataFrame accessors for petrological microprobe data."""

from __future__ import annotations

import warnings

import pandas as pd

import petropandas._calc as _calc
from petropandas._core import _is_oxide as _core_is_oxide
from petropandas._core import _oxide_cols as _core_oxide_cols
from petropandas._core import _parse_ion
from petropandas._minerals import Mineral, _score_trapezoidal


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
    """Strip whitespace, apply ALIASES, fill NaN, clip negatives on oxides."""
    from petropandas._core import ALIASES

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    work = work.rename(columns=ALIASES)
    oxide_cols = _core_oxide_cols(work)
    if oxide_cols:
        work[oxide_cols] = work[oxide_cols].fillna(0).clip(lower=0)
    return work


# ---------------------------------------------------------------------------
# MineralAccessor — general tools + mineral calculations
# ---------------------------------------------------------------------------


@pd.api.extensions.register_dataframe_accessor("mineral")
class MineralAccessor:
    """General-purpose accessor for microprobe DataFrames (``df.mineral``)."""

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = _clean_df(obj) if _needs_cleanup(obj) else obj.copy()

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
        import numpy as np

        df = self._obj
        oxide_cols = self._oxide_cols()
        units = self._units()
        idx = df.index

        # -- intermediates ---------------------------------------------------
        try:
            apfu_df = mineral._raw_apfu(df, units=units)
            fe_split_ok = True
        except KeyError:
            apfu_df = _calc.to_apfu(df, n_oxygens=mineral.n_oxygens, units=units)
            fe_split_ok = False
        sf = mineral._allocate_sites(apfu_df)

        # -- 1. analytical_total ---------------------------------------------
        total = df[oxide_cols].sum(axis=1)
        ideal_low, ideal_high = mineral.analytical_total_range
        at_scores = [_score_trapezoidal(v, ideal_low, ideal_high) for v in total]

        # -- 2. cation_deviation ---------------------------------------------
        apfu_sum = apfu_df.sum(axis=1)
        if mineral.ideal_cations is not None:
            cat_scores = [
                max(0.0, 1.0 - abs(s - mineral.ideal_cations) / mineral.ideal_cations)
                for s in apfu_sum
            ]
        else:
            cat_scores = [np.nan] * len(idx)

        # -- 3. charge_balance -----------------------------------------------
        charges = {}
        for col in apfu_df.columns:
            parsed = _parse_ion(col)
            charges[col] = parsed[1] if parsed is not None else 0
        total_charge = apfu_df.mul(pd.Series(charges)).sum(axis=1)
        expected = 2.0 * mineral.n_oxygens
        residuals = (total_charge - expected).abs()
        cb_scores = [float(np.exp(-r / 0.5)) for r in residuals]

        # -- 4. fe3+_validity ------------------------------------------------
        fe3_col = "Fe{3+}"
        fe2_col = "Fe{2+}"
        if fe_split_ok and fe3_col in apfu_df.columns:
            fe_valid = (
                (apfu_df[fe3_col] >= 0).values
                if fe2_col not in apfu_df.columns
                else ((apfu_df[fe3_col] >= 0) & (apfu_df[fe2_col] >= 0)).values
            )
            fe_scores = [1.0 if v else 0.0 for v in fe_valid]
        else:
            fe_scores = [np.nan] * len(idx)

        # -- 5. site_vacancies -----------------------------------------------
        unalloc_cols = [c for c in sf.columns if c[1] == "_unallocated"]
        site_cols = [c for c in sf.columns if c[1] != "_unallocated"]
        if unalloc_cols:
            capacities = []
            for col in unalloc_cols:
                site_name = col[0]
                cap = next(
                    (
                        s["capacity"]
                        for s in mineral.site_definitions
                        if s["name"] == site_name
                    ),
                    None,
                )
                if cap is not None and cap > 0:
                    capacities.append(cap)
            if capacities:
                mean_cap = sum(capacities) / len(capacities)
                mean_unalloc = sf[unalloc_cols].mean(axis=1)
                sv_scores = [max(0.0, 1.0 - u / mean_cap) for u in mean_unalloc]
            else:
                sv_scores = [np.nan] * len(idx)
        else:
            sv_scores = [np.nan] * len(idx)

        # -- 6. leftover_cations ---------------------------------------------
        if site_cols:
            total_ions = apfu_df.sum(axis=1)
            allocated = sf[site_cols].sum(axis=1)
            safe_total = total_ions.replace(0, 1)
            leftover_frac = ((total_ions - allocated) / safe_total).clip(lower=0)
            lc_scores = [max(0.0, 1.0 - f) for f in leftover_frac]
        else:
            lc_scores = [np.nan] * len(idx)

        # -- 7. tetrahedral_fill ---------------------------------------------
        t_site_def = next(
            (s for s in mineral.site_definitions if s["name"].startswith("T")), None
        )
        if t_site_def is not None:
            t_cols = [
                c
                for c in sf.columns
                if c[0] == t_site_def["name"] and c[1] != "_unallocated"
            ]
            if t_cols:
                t_sum = sf[t_cols].sum(axis=1)
                tf_scores = [
                    _score_trapezoidal(
                        s, t_site_def["capacity"], t_site_def["capacity"], margin=0.15
                    )
                    for s in t_sum
                ]
            else:
                tf_scores = [np.nan] * len(idx)
        else:
            tf_scores = [np.nan] * len(idx)

        # -- assemble result -------------------------------------------------
        result = pd.DataFrame(
            {
                "analytical_total": at_scores,
                "cation_deviation": cat_scores,
                "charge_balance": cb_scores,
                "fe3+_validity": fe_scores,
                "site_vacancies": sv_scores,
                "leftover_cations": lc_scores,
                "tetrahedral_fill": tf_scores,
            },
            index=idx,
        )
        return result.dropna(axis=1, how="all")


# ---------------------------------------------------------------------------
# OxidesAccessor — callable, returns oxide wt% DataFrame
# ---------------------------------------------------------------------------


@pd.api.extensions.register_dataframe_accessor("oxides")
class OxidesAccessor:
    """Callable accessor (``df.oxides``).

    ``df.oxides()`` returns a copy of the DataFrame containing only
    recognised oxide columns, converted to wt% if necessary.
    """

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = _clean_df(obj) if _needs_cleanup(obj) else obj.copy()

    def __call__(self) -> pd.DataFrame:
        """Return a copy with only recognised oxide columns in wt%."""
        units = self._obj.attrs.get("petro_units", "wt%")
        if units == "wt%":
            result = self._obj
        elif units == "apfu":
            n_oxygens = self._obj.attrs.get("petro_n_oxygens")
            n_cations = self._obj.attrs.get("petro_n_cations")
            if (n_oxygens is None) == (n_cations is None):
                msg = "Cannot convert APFU to wt% without n_oxygens or n_cations"
                raise ValueError(msg)
            total = self._obj.attrs.get("petro_total")
            result = _calc.from_apfu(
                self._obj, n_oxygens=n_oxygens, n_cations=n_cations, total=total
            )
        else:
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
        result = _calc.normalize(self._obj)
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
        moles_df = _calc.convert(self._obj, "moles")
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
        result = _calc.fe2o3_to_feo(self._obj)
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
        result = _calc.apatite_correction(self._obj)
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
class MolesAccessor:
    """Callable accessor (``df.moles``).

    ``df.moles()`` returns a copy with oxide columns as molar proportions.
    Auto-converts from the current ``petro_units`` (default: wt%).
    """

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = _clean_df(obj) if _needs_cleanup(obj) else obj.copy()

    def __call__(self) -> pd.DataFrame:
        """Return oxide columns as molar proportions."""
        units = self._obj.attrs.get("petro_units", "wt%")
        if units == "moles":
            cols = _core_oxide_cols(self._obj)
            result = self._obj[cols].copy()
        elif units == "apfu":
            n_oxygens = self._obj.attrs.get("petro_n_oxygens")
            n_cations = self._obj.attrs.get("petro_n_cations")
            if (n_oxygens is None) == (n_cations is None):
                msg = "Cannot convert APFU to moles without n_oxygens or n_cations"
                raise ValueError(msg)
            total = self._obj.attrs.get("petro_total")
            oxide_wt = _calc.from_apfu(
                self._obj, n_oxygens=n_oxygens, n_cations=n_cations, total=total
            )
            result = _calc.convert(oxide_wt, "moles")
        else:
            result = _calc.convert(self._obj, "moles", from_unit=units)
        result.attrs["petro_units"] = "moles"
        return result

    def normalized(self) -> pd.DataFrame:
        """Normalize moles to 100 mol%."""
        result = _calc.normalize(self())
        result.attrs["petro_units"] = "moles"
        return result


# ---------------------------------------------------------------------------
# ApfuAccessor — callable, returns APFU DataFrame
# ---------------------------------------------------------------------------


@pd.api.extensions.register_dataframe_accessor("apfu")
class ApfuAccessor:
    """Callable accessor (``df.apfu``).

    ``df.apfu(n_oxygens=N)`` or ``df.apfu(n_cations=N)`` returns atoms
    per formula unit.  Auto-converts from the current ``petro_units``
    (default: wt%).
    """

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = _clean_df(obj) if _needs_cleanup(obj) else obj.copy()

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
class BulkAccessor:
    """Bulk-rock geochemistry (``df.bulk``).

    ``df.bulk()`` returns a cleaned copy of the DataFrame in wt%.
    Provides normative mineralogy, alumina saturation, and oxide
    ratio calculations for whole-rock major-oxide data.
    """

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = _clean_df(obj) if _needs_cleanup(obj) else obj.copy()

    def __call__(self) -> pd.DataFrame:
        """Return a cleaned copy of the DataFrame in wt%."""
        return self._obj.copy()

    def cipw_simple(self) -> pd.DataFrame:
        """Compute CIPW normative mineralogy (simple version).

        Requires major oxides in wt% (at minimum SiO₂, Al₂O₃,
        Fe₂O₃, FeO, MgO, CaO, Na₂O, K₂O).  Missing optional
        oxides default to 0.

        Returns:
            DataFrame with normative mineral columns (Qz, Or, Ab,
            An, Di, Hy, Mt, Il, Ap, etc.) in wt%.
        """
        return _calc.cipw_norm_simple(self._obj)

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
            self._obj,
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
            self._obj,
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
        return _calc.oxide_ratios(self._obj)

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

        cols = _core_oxide_cols(self._obj)

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
        work = _calc.fe2o3_to_feo(self._obj)
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

        # ---- fill any missing columns with 0.0 -----------------------------
        for col in system_cols:
            if col not in work.columns:
                work[col] = 0.0

        return work[system_cols]

    # -- THERMOCALC ----------------------------------------------------------

    def TCbulk(
        self,
        *,
        system: str = "MnNCKFMASHTO",
        oxygen: float = 0.01,
        H2O: float = -1.0,
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
        system: str = "MnNCKFMASHTO",
        oxygen: float = 0.01,
        H2O: float = -1.0,
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
        db: str = "mp",
        sys_in: str = "mol",
        oxygen: float = 0.01,
        H2O: float = -1.0,
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
