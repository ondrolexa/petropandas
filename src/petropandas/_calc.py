"""Pure calculation functions for EMPA oxide wt% data."""

from __future__ import annotations

import pandas as pd

from petropandas._core import (
    MW,
    _cations_per,
    _detect_col,
    _element_charge,
    _element_of,
    _ion_name,
    _ion_to_oxide,
    _is_oxide,
    _oxide_cols,
    _oxygens_per,
    _parse_ion,
)


def molecular_weights(cols: pd.Index | list[str]) -> pd.Series:
    """Return molecular weight for each oxide column name.

    Args:
        cols: List or index of oxide formula strings.

    Returns:
        Series of molecular weights keyed by column name.
    """

    return pd.Series({c: MW(c) for c in cols}, dtype=float)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def _get_moles(df: pd.DataFrame, units: str) -> pd.DataFrame:
    """Return molar proportions from either wt% or mole data.

    Args:
        df: DataFrame with oxide columns.
        units: Current units (``"wt%"`` or ``"moles"``).
    """
    cols = _oxide_cols(df)
    if units == "moles":
        return df[cols]
    return to_moles(df)


def to_moles(df: pd.DataFrame) -> pd.DataFrame:
    """Convert oxide wt% to moles by dividing by molecular weight.

    Args:
        df: DataFrame with oxide columns in wt%.

    Returns:
        DataFrame of molar proportions.
    """
    cols = _oxide_cols(df)
    mw = molecular_weights(cols)
    return df[cols].div(mw)


def to_oxides(df: pd.DataFrame) -> pd.DataFrame:
    """Convert molar proportions back to oxide wt%.

    Args:
        df: DataFrame with oxide columns in moles.

    Returns:
        DataFrame of oxide weight percentages.
    """
    cols = _oxide_cols(df)
    mw = molecular_weights(cols)
    return df[cols].mul(mw)


_VALID_UNITS = {"wt%", "moles", "apfu"}


def _resolve_apfu_params(
    df: pd.DataFrame,
    n_oxygens: int | float | None,
    n_cations: int | float | None,
) -> tuple[int | float | None, int | float | None]:
    """Resolve exactly one of n_oxygens/n_cations, falling back to df.attrs.

    Args:
        df: DataFrame whose ``attrs`` may hold ``petro_n_oxygens``/
            ``petro_n_cations`` fallback values.
        n_oxygens: Explicit oxygen denominator, or *None*.
        n_cations: Explicit cation total, or *None*.

    Returns:
        The resolved ``(n_oxygens, n_cations)`` pair, with exactly one set.

    Raises:
        ValueError: If neither or both are resolvable.
    """
    if (n_oxygens is None) == (n_cations is None):
        n_oxygens = n_oxygens or df.attrs.get("petro_n_oxygens")
        n_cations = n_cations or df.attrs.get("petro_n_cations")
    if (n_oxygens is None) == (n_cations is None):
        msg = (
            "Specify exactly one of n_oxygens or n_cations "
            "(or set petro_n_oxygens/petro_n_cations in attrs)"
        )
        raise ValueError(msg)
    return n_oxygens, n_cations


def convert(
    df: pd.DataFrame,
    to_unit: str,
    *,
    from_unit: str | None = None,
    n_oxygens: int | float | None = None,
    n_cations: int | float | None = None,
) -> pd.DataFrame:
    """Convert a DataFrame between unit systems (wt%, moles, APFU).

    This is the single entry-point for all unit conversions.

    Args:
        df: DataFrame with oxide or ion columns.
        to_unit: Target unit — ``"wt%"``, ``"moles"``, or ``"apfu"``.
        from_unit: Current unit of *df*.  When *None* (default) assumed
            to be ``"wt%"``.
        n_oxygens: Oxygen denominator for APFU normalisation.
            Required when *to_unit* or *from_unit* is ``"apfu"``
            (unless found in ``df.attrs``).
        n_cations: Cation total for APFU normalisation.
            Mutually exclusive with *n_oxygens*.

    Returns:
        Converted DataFrame.

    Raises:
        ValueError: If an invalid unit is given, both or neither of
            *n_oxygens*/*n_cations* are provided when needed, or APFU
            normalisation parameters cannot be resolved.
    """
    if to_unit not in _VALID_UNITS:
        msg = f"Invalid to_unit {to_unit!r} (expected one of {sorted(_VALID_UNITS)})"
        raise ValueError(msg)

    if from_unit is None:
        from_unit = "wt%"
    elif from_unit not in _VALID_UNITS:
        msg = (
            f"Invalid from_unit {from_unit!r} (expected one of {sorted(_VALID_UNITS)})"
        )
        raise ValueError(msg)

    if from_unit == to_unit:
        return df.copy()

    # ---- wt% ↔ moles --------------------------------------------------------
    if from_unit == "wt%" and to_unit == "moles":
        return to_moles(df)
    if from_unit == "moles" and to_unit == "wt%":
        return to_oxides(df)

    # ---- → apfu -------------------------------------------------------------
    if to_unit == "apfu":
        n_oxygens, n_cations = _resolve_apfu_params(df, n_oxygens, n_cations)
        return to_apfu(df, n_oxygens=n_oxygens, n_cations=n_cations, units=from_unit)

    # ---- apfu → … -----------------------------------------------------------
    if from_unit == "apfu":
        n_oxygens, n_cations = _resolve_apfu_params(df, n_oxygens, n_cations)
        oxide_wt = from_apfu(df, n_oxygens=n_oxygens, n_cations=n_cations)
        if to_unit == "wt%":
            return oxide_wt
        # to_unit == "moles"
        return to_moles(oxide_wt)

    # unreachable
    msg = f"Unsupported conversion: {from_unit!r} → {to_unit!r}"
    raise ValueError(msg)


def cation_moles(df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
    """Compute moles of cation for each oxide.

    Args:
        df: DataFrame with oxide columns.
        units: Current units (``"wt%"`` or ``"moles"``).

    Returns:
        DataFrame of cation moles (moles × cations per formula unit).
    """
    cols = _oxide_cols(df)
    moles = _get_moles(df, units)
    cations_per = pd.Series({c: _cations_per(c) for c in cols}, dtype=float)
    return moles.mul(cations_per)


def oxygen_moles(df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
    """Compute moles of oxygen contributed by each oxide.

    Args:
        df: DataFrame with oxide columns.
        units: Current units (``"wt%"`` or ``"moles"``).

    Returns:
        DataFrame of oxygen moles (moles × oxygens per formula unit).
    """
    cols = _oxide_cols(df)
    moles = _get_moles(df, units)
    oxygens_per = pd.Series({c: _oxygens_per(c) for c in cols}, dtype=float)
    return moles.mul(oxygens_per)


# ---------------------------------------------------------------------------
# APFU — output uses ion column names
# ---------------------------------------------------------------------------


def _oxide_to_ion_col(oxide: str) -> str:
    """Convert an oxide column name to ion notation.

    Args:
        oxide: Oxide formula string (e.g. ``"SiO2"``).

    Returns:
        Ion column name (e.g. ``"Si{4+}"``).
    """
    el = _element_of(oxide)
    if not el:
        return oxide
    n_c = _cations_per(oxide)
    n_o = _oxygens_per(oxide)
    charge = 2 * n_o // n_c
    return _ion_name(el, int(charge))


def to_apfu(
    df: pd.DataFrame,
    *,
    n_oxygens: int | float | None = None,
    n_cations: int | float | None = None,
    units: str = "wt%",
) -> pd.DataFrame:
    """Convert oxide data to atoms per formula unit (APFU).

    Exactly one of ``n_oxygens`` or ``n_cations`` must be provided.

    Args:
        df: DataFrame with oxide columns.
        n_oxygens: Target number of oxygens per formula unit.
        n_cations: Target number of cations per formula unit.
        units: Current units of *df* (``"wt%"`` or ``"moles"``).

    Returns:
        DataFrame with ion-named columns (e.g. ``"Si{4+}"``, ``"Fe{2+}"``).

    Raises:
        ValueError: If both or neither of *n_oxygens*/*n_cations* are given.
    """
    if (n_oxygens is None) == (n_cations is None):
        msg = "Specify exactly one of n_oxygens or n_cations"
        raise ValueError(msg)

    cols = _oxide_cols(df)
    moles = _get_moles(df, units)
    cations_per = pd.Series({c: _cations_per(c) for c in cols}, dtype=float)
    cat = moles.mul(cations_per)

    if n_oxygens is not None:
        oxygens_per = pd.Series({c: _oxygens_per(c) for c in cols}, dtype=float)
        oxy = moles.mul(oxygens_per)
        factor = n_oxygens / oxy.sum(axis=1)
    else:
        factor = n_cations / cat.sum(axis=1)

    oxide_apfu = cat.mul(factor, axis=0)

    rename = {col: _oxide_to_ion_col(col) for col in oxide_apfu.columns}
    return oxide_apfu.rename(columns=rename)


def to_apfu_by_charge(
    df: pd.DataFrame, *, target_charges: float, units: str = "wt%"
) -> pd.DataFrame:
    """Convert oxide data to charge-normalised cation moles.

    Used for minerals normalised on total positive charge (e.g.
    chlorite's 28-charge convention) rather than a fixed oxygen count.
    Columns stay oxide-named (e.g. ``"FeO"``); callers rename to ion
    notation according to their own charge convention, since minerals
    using this convention may assign a cation's charge differently than
    the oxide it was reported as (e.g. always Fe²⁺ regardless of input).

    Args:
        df: DataFrame with oxide columns.
        target_charges: Target total positive charge per formula unit.
        units: Current units of *df* (``"wt%"`` or ``"moles"``).

    Returns:
        DataFrame of charge-normalised cation moles, oxide-named columns.
    """
    oxide_df = df if units == "wt%" else to_oxides(df)
    cols = _oxide_cols(oxide_df)
    mw = molecular_weights(cols)
    moles = oxide_df[cols].div(mw)

    charge_per = pd.Series(
        {c: _cations_per(c) * _element_charge(_element_of(c)) for c in cols}
    )
    total_charges = moles.mul(charge_per).sum(axis=1)
    factor = target_charges / total_charges

    cations_per = pd.Series({c: _cations_per(c) for c in cols}, dtype=float)
    return moles.mul(cations_per).mul(factor, axis=0)


def from_apfu(
    apfu_df: pd.DataFrame,
    *,
    n_oxygens: int | float | None = None,
    n_cations: int | float | None = None,
    total: float | None = None,
) -> pd.DataFrame:
    """Convert APFU back to oxide wt%.

    Reverses :func:`to_apfu` by mapping each ion to its standard oxide,
    computing proportional moles, normalising to *n_oxygens* or
    *n_cations*, and converting to weight percent.

    Exactly one of ``n_oxygens`` or ``n_cations`` must be provided.

    Args:
        apfu_df: DataFrame with ion-named columns (e.g. ``"Si{4+}"``).
        n_oxygens: Number of oxygens in the formula unit.
        n_cations: Number of cations in the formula unit.
        total: If given, normalise each row so the oxide wt% sum to this
            value (e.g. the original analytical total).  When *None* the
            raw formula-unit masses are returned.

    Returns:
        DataFrame with oxide columns in wt%.

    Raises:
        ValueError: If both or neither of *n_oxygens*/*n_cations* are given.
    """
    if (n_oxygens is None) == (n_cations is None):
        msg = "Specify exactly one of n_oxygens or n_cations"
        raise ValueError(msg)

    oxide_data: dict[str, pd.Series] = {}
    for col in apfu_df.columns:
        parsed = _parse_ion(col)
        if parsed is None:
            continue
        el, charge = parsed
        oxide = _ion_to_oxide(el, charge)
        cat_per = 1 if charge % 2 == 0 else 2
        oxy_per = charge // 2 if charge % 2 == 0 else charge
        prop_moles = apfu_df[col] / cat_per
        oxide_data[oxide] = {
            "prop_moles": prop_moles,
            "cat_per": cat_per,
            "oxy_per": oxy_per,
            "mw": MW(oxide),
        }

    if not oxide_data:
        return pd.DataFrame(index=apfu_df.index)

    prop_df = pd.DataFrame(
        {k: v["prop_moles"] for k, v in oxide_data.items()}, index=apfu_df.index
    )
    cat_per_series = pd.Series(
        {k: v["cat_per"] for k, v in oxide_data.items()}, dtype=float
    )
    oxy_per_series = pd.Series(
        {k: v["oxy_per"] for k, v in oxide_data.items()}, dtype=float
    )
    mw_series = pd.Series({k: v["mw"] for k, v in oxide_data.items()}, dtype=float)

    if n_oxygens is not None:
        total_ref = prop_df.mul(oxy_per_series).sum(axis=1)
    else:
        total_ref = prop_df.mul(cat_per_series).sum(axis=1)

    safe_total = total_ref.replace(0, 1)
    ref = n_oxygens if n_oxygens is not None else n_cations
    factor = ref / safe_total

    oxide_moles = prop_df.mul(factor, axis=0)
    oxide_wt = oxide_moles.mul(mw_series)

    if total is not None:
        row_totals = oxide_wt.sum(axis=1)
        safe_totals = row_totals.replace(0, 1)
        oxide_wt = oxide_wt.div(safe_totals, axis=0).mul(total, axis=0)

    return oxide_wt


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise oxide wt% so each row sums to 100%.

    Args:
        df: DataFrame with oxide columns in wt%.

    Returns:
        Normalised DataFrame.
    """
    cols = _oxide_cols(df)
    total = df[cols].sum(axis=1)
    return df[cols].div(total, axis=0) * 100.0


# ---------------------------------------------------------------------------
# Iron oxide interconversion (FeO ↔ Fe₂O₃)
# ---------------------------------------------------------------------------


def feo_to_fe2o3(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FeO wt% to Fe₂O₃ wt%, merging if Fe₂O₃ already exists.

    The FeO column is always dropped.  If Fe₂O₃ is already present, the
    converted values are **added** to it.

    Args:
        df: DataFrame with oxide columns in wt%.

    Returns:
        Copy with FeO replaced by Fe₂O₃.
    """

    work = df.copy()
    if "FeO" not in work.columns:
        return work

    feo_mw = MW("FeO")
    fe2o3_mw = MW("Fe2O3")

    feo_moles = work["FeO"] / feo_mw
    fe2o3_from_feo = feo_moles * 0.5 * fe2o3_mw

    if "Fe2O3" in work.columns:
        work["Fe2O3"] = work["Fe2O3"] + fe2o3_from_feo
    else:
        work["Fe2O3"] = fe2o3_from_feo

    work = work.drop(columns=["FeO"])
    return work


def fe2o3_to_feo(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Fe₂O₃ wt% to FeO wt%, merging if FeO already exists.

    The Fe₂O₃ column is always dropped.  If FeO is already present, the
    converted values are **added** to it.

    Args:
        df: DataFrame with oxide columns in wt%.

    Returns:
        Copy with Fe₂O₃ replaced by FeO.
    """

    work = df.copy()
    if "Fe2O3" not in work.columns:
        return work

    feo_mw = MW("FeO")
    fe2o3_mw = MW("Fe2O3")

    fe2o3_moles = work["Fe2O3"] / fe2o3_mw
    feo_from_fe2o3 = fe2o3_moles * 2 * feo_mw

    if "FeO" in work.columns:
        work["FeO"] = work["FeO"] + feo_from_fe2o3
    else:
        work["FeO"] = feo_from_fe2o3

    work = work.drop(columns=["Fe2O3"])
    return work


# ---------------------------------------------------------------------------
# Valence splitting (Fe2+/Fe3+, Mn2+/Mn3+, Ti4+/Ti3+)
# ---------------------------------------------------------------------------

_VALID_METHODS = {"droop", "schumacher"}

VALENCE_PAIRS: dict[str, dict[str, object]] = {
    "Fe": {"low_charge": 2, "high_charge": 3},
    "Mn": {"low_charge": 2, "high_charge": 3},
    "Ti": {"low_charge": 4, "high_charge": 3},
}


def _droop_apfu(
    apfu_sum: pd.Series,
    total_apfu: pd.Series,
    n_oxygens: int | float,
    ideal_cations: int | float,
) -> pd.Series:
    """Estimate the higher-charge species APFU using Droop (1987).

    Args:
        apfu_sum: Total APFU per analysis (sum of all ions).
        total_apfu: APFU of the element being split.
        n_oxygens: Number of oxygens per formula unit.
        ideal_cations: Ideal cation total for the formula.

    Returns:
        APFU of the higher-charge species.
    """
    S = apfu_sum
    F = 2.0 * n_oxygens * (1.0 - ideal_cations / S)
    high = F.clip(lower=0.0)
    low = (total_apfu - high).clip(lower=0.0)
    return total_apfu - low


def _schumacher_apfu(
    apfu: pd.DataFrame,
    element: str,
    n_oxygens: int | float,
    ideal_cations: int | float,
) -> pd.Series:
    """Estimate the higher-charge species APFU using Schumacher (1991).

    Args:
        apfu: APFU DataFrame with ion or oxide column names.
        element: Element symbol to split (e.g. ``"Fe"``).
        n_oxygens: Number of oxygens per formula unit.
        ideal_cations: Ideal cation total for the formula.

    Returns:
        APFU of the higher-charge species.
    """
    apfu_sum = apfu.sum(axis=1)
    norm = ideal_cations / apfu_sum
    cat_norm = apfu.mul(norm, axis=0)

    oxy_per_cation = {}
    for col in apfu.columns:
        ion_info = _parse_ion(col)
        if ion_info is not None:
            _el, chg = ion_info
            oxy_per_cation[col] = chg / 2.0
        elif _is_oxide(col):
            n_o = _oxygens_per(col)
            n_c = _cations_per(col)
            oxy_per_cation[col] = (2.0 * n_o) / n_c / 2.0

    present = [c for c in apfu.columns if c in oxy_per_cation]
    oxy_from_cations = (
        cat_norm[present].mul(pd.Series(oxy_per_cation)[present]).sum(axis=1)
    )

    high_apfu = 2.0 * (n_oxygens - oxy_from_cations)
    high_apfu = high_apfu.clip(lower=0.0)
    high_scaled = high_apfu / norm

    total_col = _detect_col(apfu, element)
    total_apfu = apfu[total_col]

    high = high_scaled.clip(lower=0.0)
    low = (total_apfu - high).clip(lower=0.0)
    return total_apfu - low


def split_valence(
    apfu: pd.DataFrame,
    element: str,
    method: str,
    n_oxygens: int | float,
    ideal_cations: int | float,
) -> pd.DataFrame:
    """Split a total-element column into low/high charge APFU.

    Args:
        apfu: APFU DataFrame with ion-named columns from :func:`to_apfu`.
        element: Element to split (``"Fe"``, ``"Mn"``, or ``"Ti"``).
        method: Algorithm — ``"droop"`` (Droop 1987) or ``"schumacher"``
            (Schumacher 1991).
        n_oxygens: Number of oxygens per formula unit.
        ideal_cations: Ideal cation total for the formula.

    Returns:
        DataFrame with the original element column replaced by low-charge
        and high-charge ion columns.

    Raises:
        ValueError: If *element* or *method* is not recognised.
    """
    method = method.lower()
    if element not in VALENCE_PAIRS:
        options = ", ".join(sorted(VALENCE_PAIRS))
        msg = f"Unknown element {element!r} (supported: {options})"
        raise ValueError(msg)
    if method not in _VALID_METHODS:
        options = ", ".join(sorted(_VALID_METHODS))
        msg = f"Unknown method {method!r} (supported: {options})"
        raise ValueError(msg)

    pair = VALENCE_PAIRS[element]
    low_ion = _ion_name(element, pair["low_charge"])
    high_ion = _ion_name(element, pair["high_charge"])
    if low_ion in apfu.columns and high_ion in apfu.columns:
        return apfu.copy()

    total_col = _detect_col(apfu, element)
    total_apfu = apfu[total_col]
    apfu_sum = apfu.sum(axis=1)

    if method == "droop":
        high = _droop_apfu(apfu_sum, total_apfu, n_oxygens, ideal_cations)
    else:
        high = _schumacher_apfu(apfu, element, n_oxygens, ideal_cations)

    low = (total_apfu - high).clip(lower=0.0)
    high = total_apfu - low

    result = apfu.copy()
    result = result.rename(columns={total_col: low_ion})
    result.insert(
        result.columns.get_loc(low_ion) + 1,
        high_ion,
        high,
    )
    return result


def oxidize_moles(df: pd.DataFrame, o_excess: float | pd.Series) -> pd.DataFrame:
    """Split FeO into FeO and Fe2O3 based on excess oxygen (mol%).

    Uses the THERMOCALC convention where all iron starts as Fe2+ and
    excess oxygen (O) oxidizes it via ``2 FeO + O -> Fe2O3``.

    ``o_excess`` is given in **mol%** (mole percent of the total oxide
    composition).  Internally it is converted to absolute moles via
    ``o_moles = o_excess / 100 * total_moles`` before applying:

    * ``Fe3+ = 2 * o_moles``
    * ``Fe2+ = total_FeO - 2 * o_moles`` (clipped to 0)

    If both ``FeO`` and ``Fe2O3`` columns already exist the input is
    returned unchanged (idempotent, same behaviour as
    :func:`split_valence`).

    Args:
        df: Molar-proportions DataFrame containing an ``FeO`` column.
        o_excess: Mole percent of excess oxygen relative to total oxide
            moles.  Scalar or per-row Series.

    Returns:
        DataFrame with ``FeO`` (reduced) and ``Fe2O3`` (new) columns.
    """
    if "FeO" in df.columns and "Fe2O3" in df.columns and (df["Fe2O3"] > 0).any():
        return df.copy()
    if "FeO" not in df.columns:
        return df.copy()

    total_feo = df["FeO"]
    if not isinstance(o_excess, pd.Series):
        o_excess = pd.Series(o_excess, index=df.index)
    total = df.sum(axis=1)
    o_moles = o_excess / 100.0 * total
    fe3 = (2.0 * o_moles).clip(lower=0.0, upper=total_feo)

    result = df.copy()
    result["FeO"] = total_feo - fe3
    result["Fe2O3"] = fe3 / 2.0
    return result


def reduce_moles(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Fe₂O₃ moles into FeO (1 mol Fe₂O₃ → 2 mol FeO).

    The Fe₂O₃ column is always dropped.  If FeO is already present, the
    converted values are **added** to it.

    If no Fe₂O₃ column exists, returns a copy unchanged.

    Args:
        df: Molar-proportions DataFrame.

    Returns:
        Copy with Fe₂O₃ replaced by FeO.
    """
    work = df.copy()
    if "Fe2O3" not in work.columns:
        return work

    feo_from_fe2o3 = work["Fe2O3"] * 2

    if "FeO" in work.columns:
        work["FeO"] = work["FeO"] + feo_from_fe2o3
    else:
        work["FeO"] = feo_from_fe2o3

    work = work.drop(columns=["Fe2O3"])
    return work


# ---------------------------------------------------------------------------
# Bulk-rock calculations
# ---------------------------------------------------------------------------


def alumina_saturation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute alumina saturation indices (A/NK, A/CNK).

    Molar ratios based on bulk oxide wt%.

    Args:
        df: DataFrame with major-oxide wt% columns.

    Returns:
        DataFrame with ``A/NK`` and ``A/CNK`` columns.
    """

    mw_al2o3 = MW("Al2O3")
    mw_nao2 = MW("Na2O")
    mw_k2o = MW("K2O")
    mw_cao = MW("CaO")

    al2o3 = df.get("Al2O3", 0.0) / mw_al2o3
    na2o = df.get("Na2O", 0.0) / mw_nao2
    k2o = df.get("K2O", 0.0) / mw_k2o
    cao = df.get("CaO", 0.0) / mw_cao

    nk = na2o + k2o
    nk_safe = nk.replace(0, 1)
    cnk = cao + nk
    cnk_safe = cnk.replace(0, 1)

    return pd.DataFrame(
        {"A/NK": al2o3 / nk_safe, "A/CNK": al2o3 / cnk_safe},
        index=df.index,
    )


def oxide_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute common bulk-rock oxide ratios.

    Returns a DataFrame with derived ratios including Mg#, FeOT,
    total alkalis, and K/Na.  Only ratios whose required oxides
    are all present are computed; missing ratios become NaN.

    Args:
        df: DataFrame with major-oxide wt% columns.

    Returns:
        DataFrame with ratio columns.
    """
    result = pd.DataFrame(index=df.index)

    feo = df.get("FeO")
    fe2o3 = df.get("Fe2O3")
    mgo = df.get("MgO")
    nao = df.get("Na2O")
    k2o = df.get("K2O")
    sio2 = df.get("SiO2")
    cao = df.get("CaO")
    tio2 = df.get("TiO2")

    # FeOT — total iron as FeO
    if feo is not None and fe2o3 is not None:
        result["FeOT"] = feo + 0.8998 * fe2o3
    elif feo is not None:
        result["FeOT"] = feo

    # Mg# — molar Mg/(Mg+Fe²⁺)
    if mgo is not None and feo is not None:
        mw_mgo = MW("MgO")
        mw_feo = MW("FeO")
        mg_mol = mgo / mw_mgo
        fe_mol = feo / mw_feo
        total = mg_mol + fe_mol
        result["Mg#"] = mg_mol / total.replace(0, 1)

    # Total alkalis
    if nao is not None and k2o is not None:
        result["Na2O+K2O"] = nao + k2o

    # K2O/Na2O
    if k2o is not None and nao is not None:
        result["K2O/Na2O"] = k2o / nao.replace(0, 1)

    # CaO/Na2O
    if cao is not None and nao is not None:
        result["CaO/Na2O"] = cao / nao.replace(0, 1)

    # Pass-through columns
    if sio2 is not None:
        result["SiO2"] = sio2
    if tio2 is not None:
        result["TiO2"] = tio2

    return result


def apatite_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Remove CaO bound in apatite and zero P₂O₅.

    Uses true fluorapatite stoichiometry Ca₅(PO₄)₃F (Ca:P = 5:3).
    Each mole of P₂O₅ consumes 10/3 moles of CaO (≈ 3.333 mol
    CaO per mol P₂O₅).  After subtraction the corrected CaO is
    returned in wt% and P₂O₅ is set to 0.

    Useful before thermodynamic modelling where P-bearing phases
    are absent or inaccurate — leaving P in the system falsely
    binds Ca that should be available for plagioclase, garnet, or
    clinopyroxene.

    Args:
        df: DataFrame with oxide wt% columns.

    Returns:
        Corrected copy with reduced CaO and P₂O₅ = 0.
    """

    work = df.copy()
    if "P2O5" not in work.columns:
        return work

    p2o5 = work["P2O5"]
    p2o5_moles = p2o5 / MW("P2O5")

    cao_consumed = (10.0 / 3.0) * p2o5_moles  # moles of CaO

    if "CaO" in work.columns:
        cao_mw = MW("CaO")
        work["CaO"] = work["CaO"] - cao_consumed * cao_mw

    work["P2O5"] = 0.0
    return work


def cipw_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CIPW normative mineralogy from bulk oxide wt%.

    Implements the CIPW norm algorithm (Cross, Iddings, Pirsson &
    Washington, 1902; revised by Johnson et al. 1960).

    At minimum requires SiO₂, Al₂O₃, Fe₂O₃, FeO, MgO, CaO, Na₂O,
    K₂O.  Optional: TiO₂, MnO, P₂O₅, Cr₂O₃, SO₃.  Missing oxides
    default to 0.

    Args:
        df: DataFrame with oxide wt% columns.

    Returns:
        DataFrame with normative mineral columns in wt%.
    """

    # ---- 1. read wt% and convert to molar proportions -----------------------
    idx = df.index
    zeros = pd.Series(0.0, index=idx)

    ox = {
        "SiO2": df.get("SiO2", zeros),
        "TiO2": df.get("TiO2", zeros),
        "Al2O3": df.get("Al2O3", zeros),
        "Fe2O3": df.get("Fe2O3", zeros),
        "FeO": df.get("FeO", zeros),
        "MgO": df.get("MgO", zeros),
        "CaO": df.get("CaO", zeros),
        "Na2O": df.get("Na2O", zeros),
        "K2O": df.get("K2O", zeros),
        "P2O5": df.get("P2O5", zeros),
        "Cr2O3": df.get("Cr2O3", zeros),
    }

    mol: dict[str, pd.Series] = {}
    for name, vals in ox.items():
        mol[name] = vals / MW(name)

    sio2 = mol["SiO2"]
    al2o3 = mol["Al2O3"]
    fe2o3 = mol["Fe2O3"]
    feo = mol["FeO"]
    mgo = mol["MgO"]
    cao = mol["CaO"]
    nao = mol["Na2O"]
    k2o = mol["K2O"]
    p2o5 = mol["P2O5"]
    cr2o3 = mol["Cr2O3"]

    z = lambda: pd.Series(0.0, index=idx)  # noqa: E731

    # ---- 2. normative minerals (assigned in stoichiometric order) -----------

    # Apatite  — 3CaO·P₂O₅ (simplified)
    ap = p2o5.copy()
    cao = cao - 3 * ap

    # Ilmenite  FeO·TiO₂
    il = mol["TiO2"].copy()
    feo = feo - il

    # Magnetite  FeO·Fe₂O₃
    mt = fe2o3.copy()
    feo = feo - mt
    fe2o3 = z()

    # Chromite  FeO·Cr₂O₃
    cr = cr2o3.copy()
    feo = feo - cr

    # Orthoclase  K₂O·Al₂O₃·6SiO₂
    or_ = k2o.copy()
    al2o3 = al2o3 - or_
    sio2 = sio2 - 6 * or_
    k2o = z()

    # Albite  Na₂O·Al₂O₃·6SiO₂
    ab = nao.copy()
    al2o3 = al2o3 - ab
    sio2 = sio2 - 6 * ab
    nao = z()

    # Anorthite  CaO·Al₂O₃·2SiO₂
    an = al2o3.clip(lower=0)
    an = an.clip(upper=cao)
    al2o3 = al2o3 - an
    sio2 = sio2 - 2 * an
    cao = cao - an

    # Corundum  Al₂O₃ (excess aluminium)
    c = al2o3.clip(lower=0)

    # Diopside  CaO·MgO·2SiO₂ = CaMgSi₂O₆
    di = cao.clip(lower=0).clip(upper=mgo.clip(lower=0))
    cao = cao - di
    mgo = mgo - di
    sio2 = sio2 - 2 * di

    # Hypersthene  (Mg,Fe)O·SiO₂
    hy_mg = mgo.clip(lower=0)
    hy_fe = feo.clip(lower=0)
    hy_total = hy_mg + hy_fe
    hy_total = hy_total.clip(upper=sio2.clip(lower=0))
    scale = hy_total / hy_total.replace(0, 1)
    hy_mg = hy_mg * scale
    hy_fe = hy_fe * scale
    sio2 = sio2 - hy_mg - hy_fe
    mgo = mgo - hy_mg
    feo = feo - hy_fe

    # Quartz  SiO₂ (excess silica)
    qz = sio2.clip(lower=0)

    # ---- 3. convert normative moles → wt% -----------------------------------

    def _to_wt(moles: pd.Series, formula: str) -> pd.Series:
        return moles * MW(formula)

    norm_wt: dict[str, pd.Series] = {}
    norm_wt["Ap"] = _to_wt(ap, "Ca3(PO4)2")
    norm_wt["Il"] = _to_wt(il, "FeTiO3")
    norm_wt["Mt"] = _to_wt(mt, "Fe3O4")
    norm_wt["Cr"] = _to_wt(cr, "FeCr2O4")
    norm_wt["Or"] = _to_wt(or_, "K2Al2Si6O16")
    norm_wt["Ab"] = _to_wt(ab, "Na2Al2Si6O16")
    norm_wt["An"] = _to_wt(an, "CaAl2Si2O8")
    norm_wt["C"] = _to_wt(c, "Al2O3")
    norm_wt["Di"] = _to_wt(di, "CaMgSi2O6")
    norm_wt["Hy"] = hy_mg * MW("MgSiO3") + hy_fe * MW("FeSiO3")
    norm_wt["Qz"] = _to_wt(qz, "SiO2")

    result = pd.DataFrame(norm_wt, index=idx)

    # drop zero-only columns
    result = result.loc[:, (result != 0).any(axis=0)]

    return result
