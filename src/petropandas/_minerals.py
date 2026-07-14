"""Mineral classes for structural formulas and end-member calculations.

Mineral instances are configuration objects passed to accessor methods::

    garnet = Garnet()
    df.mineral.apfu(garnet)
    df.mineral.site_allocations(garnet)
    df.mineral.end_members(garnet)
    df.mineral.check_stoichiometry(garnet)

Each mineral defines its oxygen count, ideal cation total, site
occupancy rules, valence-splitting method, and end-member algorithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import petropandas._calc as _calc


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


def _score_trapezoidal(value, ideal_low, ideal_high, margin=1.5):
    """Score *value* on a 0-1 trapezoidal scale.

    Returns 1.0 when *value* is inside ``[ideal_low, ideal_high]``,
    linearly decays to 0.0 at ``ideal_low - margin`` or
    ``ideal_high + margin``, and 0.0 outside the acceptable range.

    Args:
        value: Measured value to score.
        ideal_low: Lower bound of the perfect range.
        ideal_high: Upper bound of the perfect range.
        margin: Width of the linear decay zone on each side.

    Returns:
        Score between 0.0 and 1.0.
    """
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return max(0.0, (value - (ideal_low - margin)) / margin)
    return max(0.0, ((ideal_high + margin) - value) / margin)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Mineral:
    """Base class for mineral structural-formula calculations.

    Subclasses must set the following class attributes:

    Attributes:
        name: Human-readable mineral name.
        n_oxygens: Number of oxygens in the formula unit.
        ideal_cations: Ideal cation total, or None to omit.
        valence_splits: List of dicts with ``"element"`` and ``"method"``
            keys, optionally ``"n_oxygens"`` and ``"ideal_cations"``.
        site_definitions: Ordered list of site dicts, each with
            ``"name"``, ``"capacity"``, and ``"priority"`` (list of ion
            names such as ``"Fe{2+}"``, ``"Si{4+}"``).
        analytical_total_range: Tuple of (low, high) ideal oxide wt% sum.
    """

    name: str = "Unknown"
    n_oxygens: int | float = 0
    ideal_cations: int | float | None = None
    valence_splits: list[dict] = []
    site_definitions: list[dict] = []
    analytical_total_range: tuple[float, float] = (98.5, 101.5)

    #: Keyword arguments accepted by ``Mineral(**kwargs)`` — kept in sync
    #: with the class attributes above so a typo raises instead of
    #: silently creating a stray attribute.
    _FIELDS = frozenset(
        {
            "name",
            "n_oxygens",
            "ideal_cations",
            "valence_splits",
            "site_definitions",
            "analytical_total_range",
        }
    )

    def __init__(self, **kwargs: object) -> None:
        unknown = set(kwargs) - self._FIELDS
        if unknown:
            msg = f"Unknown Mineral field(s): {sorted(unknown)}"
            raise TypeError(msg)
        for key, val in kwargs.items():
            setattr(self, key, val)

    # -- public API ----------------------------------------------------------

    def _preprocess_oxides(self, oxide_df: pd.DataFrame) -> pd.DataFrame:
        """Hook for mineral-specific oxide preprocessing before APFU conversion.

        Default is identity. Override in subclasses that need e.g. FeO/Fe2O3
        forcing (Epidote, Titanite) or Fe2O3-into-FeO merging (Spinel)
        before the standard oxygen-normalised APFU conversion.

        Args:
            oxide_df: DataFrame with oxide columns in wt%.

        Returns:
            Preprocessed copy with oxide columns in wt%.
        """
        return oxide_df

    def _raw_apfu(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Compute raw element APFU with valence splits and ion-named columns.

        This is the internal APFU calculation before site allocation.
        Use :meth:`apfu` for site-filtered APFU or
        :meth:`site_allocations` for full site details.

        Args:
            df: DataFrame with oxide columns.
            units: Current units (``"wt%"`` or ``"moles"``).

        Returns:
            DataFrame with ion-named columns (e.g. ``"Si{4+}"``, ``"Fe{2+}"``).
        """
        oxide_df = df if units == "wt%" else _calc.to_oxides(df)
        work = self._preprocess_oxides(oxide_df)
        apfu_df = _calc.to_apfu(work, n_oxygens=self.n_oxygens, units="wt%")
        return self._apply_valence_splits(apfu_df)

    @staticmethod
    def _col(df: pd.DataFrame, name: str) -> pd.Series:
        """Return column *name* from *df*, or a zero-filled Series if absent."""
        return df[name] if name in df.columns else pd.Series(0.0, index=df.index)

    def apfu(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Compute element APFU from site allocations, excluding remainder.

        Only cations that were assigned to crystallographic sites are
        included.  Cations not in any site's priority list or in excess
        of site capacity are excluded.

        Use ``df.apfu(n_oxygens=...)`` for raw APFU before site filtering.

        Args:
            df: DataFrame with oxide columns.
            units: Current units (``"wt%"`` or ``"moles"``).

        Returns:
            DataFrame with ion-named columns (e.g. ``"Si{4+}"``, ``"Fe{2+}"``).
        """
        sf = self.site_allocations(df, units)
        cation_cols = [c for c in sf.columns if c[1] != "_unallocated"]
        if not cation_cols:
            return pd.DataFrame(index=sf.index)
        return sf[cation_cols].T.groupby(level=1).sum().T

    def site_allocations(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Compute site allocations with hierarchical (site, cation) columns.

        Args:
            df: DataFrame with oxide columns.
            units: Current units (``"wt%"`` or ``"moles"``).

        Returns:
            DataFrame with MultiIndex columns ``(site, cation)`` (e.g.
            ``("Z", "Si{4+}")``, ``("X", "Fe{2+}")``) and
            ``(site, "_unallocated")`` columns.
        """
        elem_apfu = self._raw_apfu(df, units)
        return self._allocate_sites(elem_apfu)

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Compute end-member proportions as percentages.

        Base implementation raises; subclasses that support end-member
        calculations override this method.

        Args:
            df: DataFrame with oxide columns.
            units: Current units (``"wt%"`` or ``"moles"``).

        Returns:
            DataFrame of end-member percentages.

        Raises:
            NotImplementedError: If the subclass does not define end-members.
        """
        msg = f"{type(self).__name__} does not define end-member calculations"
        raise NotImplementedError(msg)

    # -- internal helpers ----------------------------------------------------

    def _apply_valence_splits(self, apfu: pd.DataFrame) -> pd.DataFrame:
        """Apply each valence split in sequence to the APFU DataFrame."""
        result = apfu.copy()
        for split in self.valence_splits:
            element = split["element"]
            method = split["method"]
            n_oxy = split.get("n_oxygens", self.n_oxygens)
            ideal_cat = split.get("ideal_cations", self.ideal_cations)
            if ideal_cat is None:
                msg = (
                    f"ideal_cations must be set on the mineral or in "
                    f"the split dict for element {element!r}"
                )
                raise ValueError(msg)
            try:
                result = _calc.split_valence(
                    result,
                    element=element,
                    method=method,
                    n_oxygens=n_oxy,
                    ideal_cations=ideal_cat,
                )
            except KeyError:
                pass
        return result

    def _allocate_sites(self, elem_apfu: pd.DataFrame) -> pd.DataFrame:
        """Sequentially allocate element APFU to crystallographic sites.

        Sites are processed in declaration order.  Each site consumes from
        a shared pool in the order given by its ``priority`` list.

        Returns:
            DataFrame with MultiIndex columns ``(site, cation)``.
        """
        pool = elem_apfu.copy()
        result_parts: dict[tuple[str, str], pd.Series] = {}

        for site_def in self.site_definitions:
            site_name = site_def["name"]
            capacity = site_def["capacity"]
            site_remaining = pd.Series(capacity, index=pool.index, dtype=float)

            for ion in site_def["priority"]:
                if ion not in pool.columns:
                    continue
                take = pool[ion].clip(upper=site_remaining)
                result_parts[(site_name, ion)] = take
                pool[ion] = pool[ion] - take
                site_remaining = site_remaining - take

            result_parts[(site_name, "_unallocated")] = site_remaining

        return pd.DataFrame(result_parts, index=elem_apfu.index)


# ---------------------------------------------------------------------------
# Garnet — X₃Y₂Z₃O₁₂  (12 oxygens, 8 cations)
# ---------------------------------------------------------------------------


class Garnet(Mineral):
    """Garnet group — 12 oxygens, 8 cations.

    Sites: Z(3.0), Y(2.0), X(3.0).
    """

    name = "Garnet"
    n_oxygens = 12
    ideal_cations = 8
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "Z", "capacity": 3.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "Y",
            "capacity": 2.0,
            "priority": ["Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{3+}"],
        },
        {
            "name": "X",
            "capacity": 3.0,
            "priority": ["Fe{2+}", "Mg{2+}", "Ca{2+}", "Mn{2+}"],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Garnet end-member proportions via Locock (2008) sequential allocation (%).

        Uses Droop (1987) Fe³⁺ estimate and allocates end-members in order:
        1. Uvarovite — Cr into Y-site, paired with Ca
        2. Andradite — Fe³⁺ into Y-site, paired with Ca
        3. Grossular — remaining Ca into Y-site with Al
        4. Pyrope, Almandine, Spessartine — remaining Mg, Fe²⁺, Mn on X-site

        All fractions normalized to sum to 100%.
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        ca = self._col(elem_apfu, "Ca{2+}").clip(lower=0)
        mg = self._col(elem_apfu, "Mg{2+}").clip(lower=0)
        fe2 = self._col(elem_apfu, "Fe{2+}").clip(lower=0)
        fe3 = self._col(elem_apfu, "Fe{3+}").clip(lower=0)
        mn = self._col(elem_apfu, "Mn{2+}").clip(lower=0)
        cr = self._col(elem_apfu, "Cr{3+}").clip(lower=0)

        # 1. Uvarovite: Ca₃Cr₂Si₃O₁₂ → Uv = Cr/2
        uvr = (cr / 2.0).clip(upper=ca / 3.0)
        ca_rem = (ca - 3.0 * uvr).clip(lower=0)

        # 2. Andradite: Ca₃Fe³⁺₂Si₃O₁₂ → And = Fe³⁺/2
        adr = (fe3 / 2.0).clip(upper=ca_rem / 3.0)
        ca_rem = (ca_rem - 3.0 * adr).clip(lower=0)

        # 3. Grossular: Ca₃Al₂Si₃O₁₂ → Grs = Ca_rem/3
        grs = (ca_rem / 3.0).clip(lower=0)

        # 4. X-site: Pyrope, Almandine, Spessartine
        prp = (mg / 3.0).clip(lower=0)
        alm = (fe2 / 3.0).clip(lower=0)
        sps = (mn / 3.0).clip(lower=0)

        # Normalize to 100%
        total = uvr + adr + grs + prp + alm + sps
        total_safe = total.replace(0, 1)

        result = pd.DataFrame(index=idx)
        result["Prp"] = (prp / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Alm"] = (alm / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Sps"] = (sps / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Grs"] = (grs / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Adr"] = (adr / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Uvr"] = (uvr / total_safe * 100).where(total > 0, 0.0).round(3)
        return result


Grt = Garnet()


# ---------------------------------------------------------------------------
# GarnetFe3 — X₃Y₂Z₃O₁₂  (12 oxygens, Fe³⁺-aware)
# ---------------------------------------------------------------------------


class GarnetFe3(Garnet):
    """Garnet with Fe³⁺/Cr³⁺ end-members — 12 oxygens, 8 cations.

    Extends :class:`Garnet` with Andradite and Uvarovite via the matrix
    inversion method (Walters 2022, Locock 2008).

    End-members: Pyrope, Almandine, Spessartine, Grossular,
    Andradite, Uvarovite.
    """

    name = "GarnetFe3"

    # Ideal moles of each cation per formula unit (7 × 6 matrix).
    _ENDMEMBER_NAMES = ["Prp", "Alm", "Sps", "Grs", "Adr", "Uvr"]
    _IDEAL_MATRIX = np.array(
        [
            # Ca  Mg  Fe   Cr  Mn  Al  Si
            [0, 3, 0, 0, 0, 2, 3],  # Pyrope
            [0, 0, 3, 0, 0, 2, 3],  # Almandine
            [0, 0, 0, 0, 3, 2, 3],  # Spessartine
            [3, 0, 0, 0, 0, 2, 3],  # Grossular
            [3, 0, 2, 0, 0, 0, 3],  # Andradite
            [3, 0, 0, 2, 0, 0, 3],  # Uvarovite
        ],
        dtype=float,
    ).T  # shape (7, 6)

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Garnet end-members via matrix inversion (%).

        Uses bulk Fe (not Fe²⁺/Fe³⁺ split) so the matrix can
        distinguish Andradite (Y=Fe³⁺) from Grossular (Y=Al).
        """
        from petropandas._core import _element_of

        oxide_df = df if units == "wt%" else _calc.to_oxides(df)
        apfu = _calc.to_apfu(oxide_df, n_oxygens=self.n_oxygens)

        # Total Fe (FeO in input = all iron as Fe²⁺ equivalent)
        fe_cols = [c for c in apfu.columns if _element_of(c) == "Fe"]
        fe_total = (
            apfu[fe_cols].sum(axis=1) if fe_cols else pd.Series(0.0, index=apfu.index)
        )

        b = np.column_stack(
            [
                self._col(apfu, "Ca{2+}").values,
                self._col(apfu, "Mg{2+}").values,
                fe_total.values,
                self._col(apfu, "Cr{3+}").values,
                self._col(apfu, "Mn{2+}").values,
                self._col(apfu, "Al{3+}").values,
                self._col(apfu, "Si{4+}").values,
            ]
        )  # shape (n, 7)

        # Solve least-squares for each row
        x, *_ = np.linalg.lstsq(self._IDEAL_MATRIX, b.T, rcond=None)  # (6, n)
        x = np.clip(x, 0, None)  # clip negatives
        col_sums = x.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        x = x / col_sums * 100.0

        result = pd.DataFrame(
            {name: np.round(x[i], 3) for i, name in enumerate(self._ENDMEMBER_NAMES)},
            index=apfu.index,
        )
        return result


GrtFe3 = GarnetFe3()


# ---------------------------------------------------------------------------
# Feldspar — T₁M O₈  (8 oxygens)
# ---------------------------------------------------------------------------


class Feldspar(Mineral):
    """Feldspar — 8 oxygens.

    Sites: T(4.0), M(1.0).
    """

    name = "Feldspar"
    n_oxygens = 8
    ideal_cations = 5
    analytical_total_range = (99.0, 101.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 4.0, "priority": ["Si{4+}", "Al{3+}"]},
        {"name": "M", "capacity": 1.0, "priority": ["Ca{2+}", "Na{+}", "K{+}"]},
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Feldspar end-member proportions from M-site (%)."""
        sf = self.site_allocations(df, units)
        m_cols = [c for c in sf.columns if c[0] == "M" and c[1] != "_unallocated"]
        m_total = sf[m_cols].sum(axis=1)

        mapping = {"Ca{2+}": "An", "Na{+}": "Ab", "K{+}": "Or"}
        result = pd.DataFrame(index=sf.index)
        for col in m_cols:
            ion = col[1]
            name = mapping.get(ion, ion)
            result[name] = (sf[col] / m_total * 100).round(3)
        return result


Fsp = Feldspar()


# ---------------------------------------------------------------------------
# Clinopyroxene — T M1 M2 O₆  (6 oxygens, 4 cations)
# ---------------------------------------------------------------------------


class Clinopyroxene(Mineral):
    """Clinopyroxene — 6 oxygens, 4 cations.

    Sites: T(2.0), M1(1.0), M2(1.0).
    Fe³⁺ estimated via Droop (1987).

    End-members (IMA sequential allocation):
    Kosmochlor (Cr-Na), Aegirine (Fe³⁺-Na), Jadeite (Al-Na),
    Ca-Tschermak, Wollastonite, Diopside, Hedenbergite,
    Enstatite, Ferrosilite.
    """

    name = "Clinopyroxene"
    n_oxygens = 6
    ideal_cations = 4
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "T", "capacity": 2.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "M1",
            "capacity": 1.0,
            "priority": ["Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{3+}", "Mg{2+}", "Fe{2+}"],
        },
        {
            "name": "M2",
            "capacity": 1.0,
            "priority": ["Ca{2+}", "Na{+}", "Mn{2+}", "Fe{2+}", "Mg{2+}"],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Clinopyroxene end-members via IMA sequential allocation (%).

        Strict allocation hierarchy (Morimoto 1988):
        1. T-site: Si → Al_IV → rest is Al_VI
        2. Sodium: Kosmochlor → Aegirine → Jadeite
        3. Tschermak: CaTs from remaining Al_VI
        4. Quad: Di, Hd, Wo, En, Fs from remaining Ca, Mg, Fe²⁺

        All fractions are normalized to sum to 100%.
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        si = self._col(elem_apfu, "Si{4+}")
        al = self._col(elem_apfu, "Al{3+}")
        cr = self._col(elem_apfu, "Cr{3+}")
        fe3 = self._col(elem_apfu, "Fe{3+}")
        fe2 = self._col(elem_apfu, "Fe{2+}")
        mg = self._col(elem_apfu, "Mg{2+}")
        ca = self._col(elem_apfu, "Ca{2+}")
        na = self._col(elem_apfu, "Na{+}")

        # --- 1. T-site: Si, then Al to fill to 2.0 ---
        al_iv = ((2.0 - si).clip(lower=0)).clip(upper=al)
        al_vi = (al - al_iv).clip(lower=0)

        # --- 2. Na end-members (IMA order) ---
        krs = na.clip(upper=cr)
        ae = (na - krs).clip(lower=0).clip(upper=fe3)
        jd = (na - krs - ae).clip(lower=0).clip(upper=al_vi)

        # --- 3. Ca-Tschermak (2 Al per formula) ---
        al_vi_rem = (al_vi - jd).clip(lower=0)
        cats = (al_vi_rem / 2.0).clip(upper=ca)
        ca_rem = (ca - cats).clip(lower=0)

        # --- 4. Quad: Di, Hd, Wo, En, Fs ---
        mg_fe_total = (mg + fe2).replace(0, 1)
        di = (ca_rem * mg / mg_fe_total).clip(upper=mg)
        hd = (ca_rem * fe2 / mg_fe_total).clip(upper=fe2)
        wo = (ca_rem - di - hd).clip(lower=0)

        en = ((mg - di).clip(lower=0)) / 2.0
        fs = ((fe2 - hd).clip(lower=0)) / 2.0

        # --- 5. Normalize to 100% ---
        raw = krs + ae + jd + cats + wo + di + hd + en + fs
        raw_safe = raw.replace(0, 1)

        result = pd.DataFrame(index=idx)

        result["Jd"] = (jd / raw_safe * 100).where(raw > 0, 0.0).round(3)
        result["Ae"] = (ae / raw_safe * 100).where(raw > 0, 0.0).round(3)
        result["Di"] = (di / raw_safe * 100).where(raw > 0, 0.0).round(3)
        result["Hd"] = (hd / raw_safe * 100).where(raw > 0, 0.0).round(3)
        result["Kosmochlor"] = (krs / raw_safe * 100).where(raw > 0, 0.0).round(3)
        result["CaTs"] = (cats / raw_safe * 100).where(raw > 0, 0.0).round(3)
        result["Wo"] = (wo / raw_safe * 100).where(raw > 0, 0.0).round(3)
        result["En"] = (en / raw_safe * 100).where(raw > 0, 0.0).round(3)
        result["Fs"] = (fs / raw_safe * 100).where(raw > 0, 0.0).round(3)
        return result


Cpx = Clinopyroxene()


# ---------------------------------------------------------------------------
# Orthopyroxene — T M1 M2 O₆  (6 oxygens, 4 cations)
# ---------------------------------------------------------------------------


class Orthopyroxene(Mineral):
    """Orthopyroxene — 6 oxygens, 4 cations.

    Sites: T(2.0), M1(1.0), M2(1.0).
    End-members: MgTs, Wo, En, Fs.
    """

    name = "Orthopyroxene"
    n_oxygens = 6
    ideal_cations = 4
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "T", "capacity": 2.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "M1",
            "capacity": 1.0,
            "priority": ["Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{3+}", "Mg{2+}", "Fe{2+}"],
        },
        {
            "name": "M2",
            "capacity": 1.0,
            "priority": ["Ca{2+}", "Mn{2+}", "Fe{2+}", "Mg{2+}"],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Orthopyroxene end-members via sequential allocation (%)."""
        elem_apfu = self._raw_apfu(df, units)
        result = pd.DataFrame(index=elem_apfu.index)
        pool = elem_apfu.copy()

        # T-site: extract Mg-Tschermak (excess Al beyond Si capacity)
        si = (
            pool["Si{4+}"].clip(lower=0)
            if "Si{4+}" in pool.columns
            else pd.Series(0.0, index=pool.index)
        )
        al = (
            pool["Al{3+}"].clip(lower=0)
            if "Al{3+}" in pool.columns
            else pd.Series(0.0, index=pool.index)
        )

        mgtsermaki = (al - (2.0 - si)).clip(lower=0)
        t_consumed_si = si.clip(upper=2.0)
        t_consumed_al = (al - mgtsermaki).clip(upper=2.0 - t_consumed_si)

        if "Si{4+}" in pool.columns:
            pool["Si{4+}"] = pool["Si{4+}"] - t_consumed_si
        if "Al{3+}" in pool.columns:
            pool["Al{3+}"] = pool["Al{3+}"] - t_consumed_al

        # M1-site
        m1_remaining = pd.Series(1.0, index=pool.index, dtype=float)
        for ion in ["Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{3+}", "Mg{2+}", "Fe{2+}"]:
            if ion not in pool.columns:
                continue
            take = pool[ion].clip(upper=m1_remaining)
            pool[ion] = pool[ion] - take
            m1_remaining = m1_remaining - take

        # M2-site
        ca = pool["Ca{2+}"].clip(lower=0) if "Ca{2+}" in pool.columns else 0.0
        mn = pool["Mn{2+}"].clip(lower=0) if "Mn{2+}" in pool.columns else 0.0
        fe2 = pool["Fe{2+}"].clip(lower=0) if "Fe{2+}" in pool.columns else 0.0
        mg = pool["Mg{2+}"].clip(lower=0) if "Mg{2+}" in pool.columns else 0.0

        wo = ca
        en_fs_cap = (1.0 - wo - mn).clip(lower=0)
        mg_fe_total = (mg + fe2).replace(0, 1)
        en = en_fs_cap * (mg / mg_fe_total)
        fs = en_fs_cap * (fe2 / mg_fe_total)

        result["MgTs"] = (mgtsermaki * 100).round(3)
        result["Wo"] = (wo * 100).round(3)
        result["En"] = (en * 100).round(3)
        result["Fs"] = (fs * 100).round(3)
        return result


Opx = Orthopyroxene()


# ---------------------------------------------------------------------------
# Muscovite — T₂I₁O₅(OH)₂  (11 oxygens)
# ---------------------------------------------------------------------------


class Muscovite(Mineral):
    """Dioctahedral white mica — 11 oxygens (22 positive charges).

    Sites: T(4.0), I(1.0), O(2.0).  Fe is assumed Fe²⁺ (no valence
    split for micas — Walters 2022).

    End-members (MinPlot algorithm, Walters 2022):
    Al-Celadonite, Fe-Al-Celadonite, Pyrophyllite, Margarite,
    Paragonite, Muscovite, Trioctahedral component.
    """

    name = "Muscovite"
    n_oxygens = 11
    ideal_cations = 7.0
    analytical_total_range = (94.0, 97.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 4.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "I",
            "capacity": 1.0,
            "priority": ["K{+}", "Na{+}", "Ca{2+}", "Ba{2+}"],
        },
        {
            "name": "O",
            "capacity": 2.0,
            "priority": ["Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{2+}", "Mg{2+}", "Mn{2+}"],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Dioctahedral mica end-members via MinPlot algorithm (%).

        Returns Al-Celadonite, Fe-Al-Celadonite, Pyrophyllite,
        Margarite, Paragonite, Muscovite, and Trioctahedral component.
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        si = self._col(elem_apfu, "Si{4+}")
        al = self._col(elem_apfu, "Al{3+}")
        ti = self._col(elem_apfu, "Ti{4+}")
        cr = self._col(elem_apfu, "Cr{3+}")
        fe = self._col(elem_apfu, "Fe{2+}")
        mn = self._col(elem_apfu, "Mn{2+}")
        mg = self._col(elem_apfu, "Mg{2+}")
        k = self._col(elem_apfu, "K{+}")
        na = self._col(elem_apfu, "Na{+}")
        ca = self._col(elem_apfu, "Ca{2+}")

        # Structural site assignment
        al_iv = (4.0 - si).clip(lower=0).clip(upper=al)
        al_vi = al - al_iv
        m_sum = al_vi + ti + cr + fe + mn + mg

        # Trioctahedral vs dioctahedral
        x_trioct = (m_sum - 2.0).clip(lower=0).clip(upper=1.0)
        x_dioct = 1.0 - x_trioct

        # Tschermak split: muscovite-like (XM) vs celadonite-like
        xm = (al_vi - 1.0).clip(lower=0).clip(upper=1.0)
        x_cel = 1.0 - xm

        # Celadonite split: Mg vs Fe
        mg_fe = mg + fe
        x_mg = (mg / mg_fe.replace(0, 1)).where(mg_fe > 0, 0.0)
        x_mgcel = x_mg * x_cel
        x_fecel = x_cel - x_mgcel

        # Pyrophyllite (interlayer vacancy)
        alkali = k + na + ca
        x_mpm = alkali * xm
        x_prl = xm - x_mpm

        # Muscovite / Paragonite / Margarite
        alkali_safe = alkali.replace(0, 1)
        x_ms = (k / alkali_safe * x_mpm).where(alkali > 0, 0.0)
        x_pg = (na / alkali_safe * x_mpm).where(alkali > 0, 0.0)
        x_mrg = (ca / alkali_safe * x_mpm).where(alkali > 0, 0.0)

        # Scale dioctahedral component
        result = pd.DataFrame(index=idx)
        result["Al-Celadonite"] = (x_mgcel * x_dioct * 100).round(3)
        result["Fe-Al-Celadonite"] = (x_fecel * x_dioct * 100).round(3)
        result["Pyrophyllite"] = (x_prl * x_dioct * 100).round(3)
        result["Margarite"] = (x_mrg * x_dioct * 100).round(3)
        result["Paragonite"] = (x_pg * x_dioct * 100).round(3)
        result["Muscovite"] = (x_ms * x_dioct * 100).round(3)
        result["Trioctahedral"] = (x_trioct * 100).round(3)
        return result


Ms = Muscovite()


# ---------------------------------------------------------------------------
# Biotite — T₂I₁O₅(OH)₂  (11 oxygens)
# ---------------------------------------------------------------------------


class Biotite(Mineral):
    """Trioctahedral mica — 11 oxygens (22 positive charges).

    Sites: T(4.0), I(1.0), O(3.0).  Fe is assumed Fe²⁺ (no valence
    split for micas — Walters 2022).

    End-members (MinPlot algorithm, Walters 2022):
    Phlogopite, Annite, Eastonite, Siderophyllite, Dioctahedral component.
    """

    name = "Biotite"
    n_oxygens = 11
    ideal_cations = 7.0
    analytical_total_range = (94.0, 97.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 4.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "I",
            "capacity": 1.0,
            "priority": ["K{+}", "Na{+}", "Ba{2+}"],
        },
        {
            "name": "O",
            "capacity": 3.0,
            "priority": ["Mg{2+}", "Fe{2+}", "Al{3+}", "Ti{4+}", "Cr{3+}", "Mn{2+}"],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Trioctahedral mica end-members via MinPlot algorithm (%).

        Returns Phlogopite, Annite, Eastonite, Siderophyllite,
        and Dioctahedral component.
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        si = self._col(elem_apfu, "Si{4+}")
        al = self._col(elem_apfu, "Al{3+}")
        ti = self._col(elem_apfu, "Ti{4+}")
        cr = self._col(elem_apfu, "Cr{3+}")
        fe = self._col(elem_apfu, "Fe{2+}")
        mn = self._col(elem_apfu, "Mn{2+}")
        mg = self._col(elem_apfu, "Mg{2+}")

        # Trioctahedral vs dioctahedral (ΣM = octahedral sum)
        al_iv = (4.0 - si).clip(lower=0).clip(upper=al)
        al_vi = al - al_iv
        m_sum = al_vi + ti + cr + fe + mn + mg
        x_trioct = (m_sum - 2.0).clip(lower=0).clip(upper=1.0)
        x_dioct = 1.0 - x_trioct

        # Phlogopite-Annite join (Si = 3) vs Siderophyllite-Eastonite join (Si = 2)
        x_phlann = (si - 2.0).clip(lower=0).clip(upper=1.0)
        x_sideast = 1.0 - x_phlann

        # Mg fraction
        mg_fe = mg + fe
        x_mg = (mg / mg_fe.replace(0, 1)).where(mg_fe > 0, 0.0)

        # End-members scaled by trioctahedral component
        x_phl = x_phlann * x_mg * x_trioct
        x_ann = (x_phlann - x_phlann * x_mg) * x_trioct
        x_eas = x_sideast * x_mg * x_trioct
        x_sid = (x_sideast - x_sideast * x_mg) * x_trioct

        result = pd.DataFrame(index=idx)
        result["Phlogopite"] = (x_phl * 100).round(3)
        result["Annite"] = (x_ann * 100).round(3)
        result["Eastonite"] = (x_eas * 100).round(3)
        result["Siderophyllite"] = (x_sid * 100).round(3)
        result["Dioctahedral"] = (x_dioct * 100).round(3)
        return result


Bt = Biotite()


# ---------------------------------------------------------------------------
# Staurolite — A₂B₄C₁₈D₄T₈O₄₈(OH,Cl)₂  (48 oxygens)
# ---------------------------------------------------------------------------


class Staurolite(Mineral):
    """Staurolite — 48 oxygens.

    Sites: T(8.0), M(12.0) — simplified.
    Fe is assumed Fe²⁺ (no valence split).

    End-members: Fe-Staurolite, Mg-Staurolite, Zn-Staurolite, Mn-Staurolite.
    """

    name = "Staurolite"
    n_oxygens = 48
    ideal_cations = None
    analytical_total_range = (99.0, 101.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 8.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "M",
            "capacity": 12.0,
            "priority": [
                "Al{3+}",
                "Ti{4+}",
                "Cr{3+}",
                "Fe{2+}",
                "Mg{2+}",
                "Mn{2+}",
                "Zn{2+}",
            ],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Staurolite end-member proportions from R²⁺ occupancy (%)."""
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        fe = self._col(elem_apfu, "Fe{2+}")
        mg = self._col(elem_apfu, "Mg{2+}")
        zn = self._col(elem_apfu, "Zn{2+}")
        mn = self._col(elem_apfu, "Mn{2+}")
        r2_total = fe + mg + zn + mn

        result = pd.DataFrame(index=idx)
        r2_safe = r2_total.replace(0, 1)
        result["Fe-Staurolite"] = (fe / r2_safe * 100).where(r2_total > 0, 0.0).round(3)
        result["Mg-Staurolite"] = (mg / r2_safe * 100).where(r2_total > 0, 0.0).round(3)
        result["Zn-Staurolite"] = (zn / r2_safe * 100).where(r2_total > 0, 0.0).round(3)
        result["Mn-Staurolite"] = (mn / r2_safe * 100).where(r2_total > 0, 0.0).round(3)
        return result


St = Staurolite()


# ---------------------------------------------------------------------------
# Chlorite — (Mg,Fe²⁺,Al)₆(Si,Al)₄O₁₀(OH)₈  (28 charges = ~14 O)
# ---------------------------------------------------------------------------


class Chlorite(Mineral):
    """Chlorite group — 28 positive charges (charge-based normalization).

    Sites: T(4.0), M(6.0).  Fe is assumed Fe²⁺ (Dubacq & Forshaw 2024).
    Uses charge-based APFU normalization (28 charges), not oxygen-based.

    End-members (MinPlot algorithm):
    Clinochlore, Chamosite, Mg-Sudoite, Fe-Sudoite.
    """

    name = "Chlorite"
    n_oxygens = 14
    ideal_cations = None
    analytical_total_range = (85.0, 90.0)
    valence_splits = []
    site_definitions = [
        {"name": "T", "capacity": 4.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "M",
            "capacity": 6.0,
            "priority": [
                "Al{3+}",
                "Ti{4+}",
                "Cr{3+}",
                "Fe{2+}",
                "Mg{2+}",
                "Mn{2+}",
            ],
        },
    ]

    def _raw_apfu(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Compute raw APFU normalized to 28 positive charges."""
        cat_moles = _calc.to_apfu_by_charge(df, target_charges=28.0, units=units)
        rename = {c: _calc._oxide_to_ion_col(c) for c in cat_moles.columns}
        return cat_moles.rename(columns=rename)

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Chlorite end-members via MinPlot algorithm (%).

        Returns Clinochlore, Chamosite, Mg-Sudoite, Fe-Sudoite.
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        si = self._col(elem_apfu, "Si{4+}")
        fe = self._col(elem_apfu, "Fe{2+}")
        mg = self._col(elem_apfu, "Mg{2+}")

        r2 = mg + fe
        x_mg = (mg / r2.replace(0, 1)).where(r2 > 0, 0.0)

        x_normal = ((si - 2.0) / 1.0).clip(lower=0).clip(upper=1.0)
        x_tsch = 1.0 - x_normal

        result = pd.DataFrame(index=idx)
        result["Clinochlore"] = (x_normal * x_mg * 100).round(3)
        result["Chamosite"] = (x_normal * (1.0 - x_mg) * 100).round(3)
        result["Mg-Sudoite"] = (x_tsch * x_mg * 100).round(3)
        result["Fe-Sudoite"] = (x_tsch * (1.0 - x_mg) * 100).round(3)
        return result


Chl = Chlorite()


# ---------------------------------------------------------------------------
# Epidote — Ca₂(Al,Fe³⁺)₃(SiO₄)(Si₂O₇)O(OH)  (12.5 oxygens, 8 cations)
# ---------------------------------------------------------------------------


class Epidote(Mineral):
    """Epidote group — 12.5 oxygens (8 cations, 25 charges).

    Sites: A(2.0), M(3.0), T(3.0).  All Fe is Fe³⁺ — input FeO is
    internally converted to Fe₂O₃ before normalization.

    End-members: Clinozoisite, Epidote, Piemontite, Mukhinite, Tawmawite.
    """

    name = "Epidote"
    n_oxygens = 12.5
    ideal_cations = 8
    analytical_total_range = (99.0, 101.0)
    valence_splits = []
    site_definitions = [
        {
            "name": "A",
            "capacity": 2.0,
            "priority": ["Ca{2+}", "Mn{2+}", "Sr{2+}"],
        },
        {
            "name": "M",
            "capacity": 3.0,
            "priority": [
                "Al{3+}",
                "Fe{3+}",
                "Ti{3+}",
                "V{3+}",
                "Cr{3+}",
                "Mn{3+}",
                "Fe{2+}",
                "Mg{2+}",
            ],
        },
        {"name": "T", "capacity": 3.0, "priority": ["Si{4+}"]},
    ]

    def _preprocess_oxides(self, oxide_df: pd.DataFrame) -> pd.DataFrame:
        """Convert Fe to Fe³⁺ (Fe₂O₃ basis) before APFU normalisation.

        EMPA reports total Fe as FeO, but epidote has Fe³⁺.
        FeO input → Fe₂O₃ internally.
        """
        return _calc.feo_to_fe2o3(oxide_df)

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Epidote end-members via M-site allocation (%).

        Clinozoisite (Al-M3), Epidote (Fe³⁺-M3), Piemontite (Mn³⁺-M3),
        Mukhinite (V³⁺-M3), Tawmawite (Cr³⁺-M3).
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        al = self._col(elem_apfu, "Al{3+}")
        fe3 = self._col(elem_apfu, "Fe{3+}")
        mn3 = self._col(elem_apfu, "Mn{3+}")
        v3 = self._col(elem_apfu, "V{3+}")
        cr3 = self._col(elem_apfu, "Cr{3+}")

        m_total = al + fe3 + mn3 + v3 + cr3
        m_safe = m_total.replace(0, 1)

        result = pd.DataFrame(index=idx)
        result["Clinozoisite"] = (al / m_safe * 100).where(m_total > 0, 0.0).round(3)
        result["Epidote"] = (fe3 / m_safe * 100).where(m_total > 0, 0.0).round(3)
        result["Piemontite"] = (mn3 / m_safe * 100).where(m_total > 0, 0.0).round(3)
        result["Mukhinite"] = (v3 / m_safe * 100).where(m_total > 0, 0.0).round(3)
        result["Tawmawite"] = (cr3 / m_safe * 100).where(m_total > 0, 0.0).round(3)
        return result


Ep = Epidote()


# ---------------------------------------------------------------------------
# Amphibole — A₀₋₁B₂C₅T₈O₂₂(OH)₂  (23 oxygens, ~15 cations)
# ---------------------------------------------------------------------------


class Amphibole(Mineral):
    """Amphibole supergroup — 23 oxygens.

    Sites: A(0–1), B(2), C(5), T(8).
    Fe³⁺ estimated via Schumacher (1991) method.

    End-members (12 most common):
    Tremolite, Actinolite, Edenite, Ferro-Edenite, Pargasite,
    Ferro-Pargasite, Tschermakite, Richterite, Winchite,
    Glaucophane, Riebeckite, Magnesio-Riebeckite.
    """

    name = "Amphibole"
    n_oxygens = 23
    ideal_cations = 15
    analytical_total_range = (96.0, 99.0)
    valence_splits = [{"element": "Fe", "method": "schumacher"}]
    site_definitions = [
        {"name": "A", "capacity": 1.0, "priority": ["K{+}", "Na{+}"]},
        {
            "name": "B",
            "capacity": 2.0,
            "priority": ["Na{+}", "Ca{2+}", "Mn{2+}", "Fe{2+}", "Mg{2+}"],
        },
        {
            "name": "C",
            "capacity": 5.0,
            "priority": [
                "Mg{2+}",
                "Fe{2+}",
                "Al{3+}",
                "Ti{4+}",
                "Cr{3+}",
                "Fe{3+}",
                "Mn{2+}",
                "Na{+}",
            ],
        },
        {"name": "T", "capacity": 8.0, "priority": ["Si{4+}", "Al{3+}"]},
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Amphibole end-members via sequential allocation (%).

        Groups: calcic, sodic-calcic, sodic.
        Within each group, binary mixing end-members are computed.
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        # Site allocation
        sf = self.site_allocations(df, units)

        def _site_ion(site: str, ion: str) -> pd.Series:
            key = (site, ion)
            return sf[key] if key in sf.columns else pd.Series(0.0, index=idx)

        # A-site
        a_total = _site_ion("A", "K{+}") + _site_ion("A", "Na{+}")

        # B-site
        na_b = _site_ion("B", "Na{+}")
        ca_b = _site_ion("B", "Ca{2+}")

        # C-site
        mg_c = _site_ion("C", "Mg{2+}")
        fe2_c = _site_ion("C", "Fe{2+}")

        # T-site
        al_t = _site_ion("T", "Al{3+}")

        # Group classification
        is_calcic = ca_b >= na_b
        is_sodic_calcic = (na_b > ca_b) & (ca_b > 0)
        is_sodic = (na_b > 0) & (ca_b == 0)

        # --- Calcic end-members ---
        mg_fe_c = (mg_c + fe2_c).replace(0, 1)
        x_mg = mg_c / mg_fe_c
        x_fe = fe2_c / mg_fe_c

        x_al_t_norm = (al_t / 8.0).clip(upper=1.0)
        x_edenite = a_total.clip(upper=1.0)
        x_tschermak = x_al_t_norm

        x_trem = is_calcic * x_mg * (1.0 - x_edenite) * (1.0 - x_tschermak)
        x_act = is_calcic * x_fe * (1.0 - x_edenite) * (1.0 - x_tschermak)
        x_eas = is_calcic * x_mg * x_edenite * (1.0 - x_tschermak)
        x_ferro_eas = is_calcic * x_fe * x_edenite * (1.0 - x_tschermak)
        x_prg = is_calcic * x_mg * x_edenite * x_tschermak
        x_ferro_prg = is_calcic * x_fe * x_edenite * x_tschermak
        x_tsch = is_calcic * (1.0 - x_edenite) * x_tschermak

        # --- Sodic-calcic end-members ---
        # Richterite: Na_B > 0, Ca_B > 0, Mg-dominant
        x_richt = is_sodic_calcic * x_mg * (1.0 - x_tschermak)
        # Winchite: Na_B > 0, Ca_B > 0, Fe-dominant
        x_winch = is_sodic_calcic * x_fe * (1.0 - x_tschermak)

        # --- Sodic end-members ---
        # Glaucophane: Mg-dominant, Al in T-site
        x_glau = is_sodic * x_mg * x_tschermak
        x_ferro_glau = is_sodic * x_fe * x_tschermak

        # Riebeckite: Fe-dominant
        x_rieb = is_sodic * x_fe * (1.0 - x_tschermak)
        x_mg_rieb = is_sodic * x_mg * (1.0 - x_tschermak)

        result = pd.DataFrame(index=idx)
        result["Tremolite"] = (x_trem * 100).round(3)
        result["Actinolite"] = (x_act * 100).round(3)
        result["Edenite"] = (x_eas * 100).round(3)
        result["Ferro-Edenite"] = (x_ferro_eas * 100).round(3)
        result["Pargasite"] = (x_prg * 100).round(3)
        result["Ferro-Pargasite"] = (x_ferro_prg * 100).round(3)
        result["Tschermakite"] = (x_tsch * 100).round(3)
        result["Richterite"] = (x_richt * 100).round(3)
        result["Winchite"] = (x_winch * 100).round(3)
        result["Glaucophane"] = (x_glau * 100).round(3)
        result["Ferro-Glaucophane"] = (x_ferro_glau * 100).round(3)
        result["Riebeckite"] = (x_rieb * 100).round(3)
        result["Magnesio-Riebeckite"] = (x_mg_rieb * 100).round(3)
        return result


Amp = Amphibole()


# ---------------------------------------------------------------------------
# Titanite — CaTiSiO₅  (5 oxygens, 3 cations)
# ---------------------------------------------------------------------------


class Titanite(Mineral):
    """Titanite (sphene) — 5 oxygens, 3 cations.

    Ideal formula: CaTi(SiO₄)O

    Sites: A(1.0), B(1.0), T(1.0).
    All Fe is treated as Fe³⁺ — input FeO is internally converted to
    Fe₂O₃ before normalization (Oberti et al. 1991; Enami et al. 1993;
    King et al. 2013).

    End-members (B-site occupancy, %):
    Ttn (Titanite), Al-Ttn (Al-titanite), Fe-Ttn (Ferro-titanite),
    Mal (Malayaite), Other (Nb, Zr, etc.).
    """

    name = "Titanite"
    n_oxygens = 5
    ideal_cations = 3
    analytical_total_range = (99.0, 101.0)
    valence_splits = []
    site_definitions = [
        {
            "name": "A",
            "capacity": 1.0,
            "priority": ["Ca{2+}", "Sr{2+}", "Mn{2+}", "Na{+}"],
        },
        {
            "name": "B",
            "capacity": 1.0,
            "priority": [
                "Ti{4+}",
                "Al{3+}",
                "Fe{3+}",
                "Sn{4+}",
                "Nb{5+}",
                "Zr{4+}",
                "Cr{3+}",
                "V{3+}",
                "Mg{2+}",
            ],
        },
        {"name": "T", "capacity": 1.0, "priority": ["Si{4+}", "P{5+}"]},
    ]

    def _preprocess_oxides(self, oxide_df: pd.DataFrame) -> pd.DataFrame:
        """Convert Fe to Fe³⁺ (Fe₂O₃ basis) before APFU normalisation.

        EMPA reports total Fe as FeO, but titanite has Fe³⁺.
        FeO input → Fe₂O₃ internally.
        """
        return _calc.feo_to_fe2o3(oxide_df)

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Titanite end-member proportions from B-site occupancy (%).

        End-members: Ttn (titanite), Al-Ttn (al-titanite),
        Fe-Ttn (ferro-titanite), Mal (malayaite), Other.
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        # B-site cations (octahedral site)
        ti = self._col(elem_apfu, "Ti{4+}")
        al = self._col(elem_apfu, "Al{3+}")
        fe3 = self._col(elem_apfu, "Fe{3+}")
        sn = self._col(elem_apfu, "Sn{4+}")
        nb = self._col(elem_apfu, "Nb{5+}")
        zr = self._col(elem_apfu, "Zr{4+}")
        cr = self._col(elem_apfu, "Cr{3+}")
        v = self._col(elem_apfu, "V{3+}")
        mg = self._col(elem_apfu, "Mg{2+}")

        b_total = ti + al + fe3 + sn + nb + zr + cr + v + mg
        b_safe = b_total.replace(0, 1)

        result = pd.DataFrame(index=idx)
        result["Ttn"] = (ti / b_safe * 100).where(b_total > 0, 0.0).round(3)
        result["Al-Ttn"] = (al / b_safe * 100).where(b_total > 0, 0.0).round(3)
        result["Fe-Ttn"] = (fe3 / b_safe * 100).where(b_total > 0, 0.0).round(3)
        result["Mal"] = (sn / b_safe * 100).where(b_total > 0, 0.0).round(3)
        other = nb + zr + cr + v + mg
        result["Other"] = (other / b_safe * 100).where(b_total > 0, 0.0).round(3)
        return result


Ttn = Titanite()


# ---------------------------------------------------------------------------
# Chloritoid — (Fe²⁺,Mg,Mn)₂Al₄Si₂O₁₀(OH)₄  (12 oxygens, 8 cations)
# ---------------------------------------------------------------------------


class Chloritoid(Mineral):
    """Chloritoid group — 12 oxygens, 8 cations.

    Sites: T(2.0), M1(6.0).
    Fe³⁺ estimated via Droop (1987).

    End-members: Cld (ferrochloritoid), Mgcld (magnesiochloritoid),
    Mncld (manganese chloritoid).
    """

    name = "Chloritoid"
    n_oxygens = 12
    ideal_cations = 8
    analytical_total_range = (99.0, 101.0)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {"name": "T", "capacity": 2.0, "priority": ["Si{4+}", "Al{3+}"]},
        {
            "name": "M1",
            "capacity": 6.0,
            "priority": [
                "Al{3+}",
                "Ti{4+}",
                "Cr{3+}",
                "Fe{3+}",
                "Fe{2+}",
                "Mg{2+}",
                "Mn{2+}",
            ],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Chloritoid end-member proportions from M1-site R²⁺ occupancy (%).

        End-members: Cld (Fe²⁺-dominant), Mgcld (Mg-dominant),
        Mncld (Mn-dominant).
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        fe2 = self._col(elem_apfu, "Fe{2+}")
        mg = self._col(elem_apfu, "Mg{2+}")
        mn = self._col(elem_apfu, "Mn{2+}")
        r2_total = fe2 + mg + mn
        r2_safe = r2_total.replace(0, 1)

        result = pd.DataFrame(index=idx)
        result["Cld"] = (fe2 / r2_safe * 100).where(r2_total > 0, 0.0).round(3)
        result["Mgcld"] = (mg / r2_safe * 100).where(r2_total > 0, 0.0).round(3)
        result["Mncld"] = (mn / r2_safe * 100).where(r2_total > 0, 0.0).round(3)
        return result


Cld = Chloritoid()


# ---------------------------------------------------------------------------
# Cordierite — (Mg,Fe)₂Al₄Si₅O₁₈  (18 oxygens, ~11 cations excl. A-site)
# ---------------------------------------------------------------------------


class Cordierite(Mineral):
    """Cordierite group — 18 oxygens, 11 cations (excl. A-site).

    Sites: T1(6.0), T2(3.0), B(2.0), A(0–1.0).
    Fe is assumed Fe²⁺ (no valence split).

    End-members: H₂O-Crd (channel filling), Mg-Crd, Fe-Crd.
    """

    name = "Cordierite"
    n_oxygens = 18
    ideal_cations = 11
    analytical_total_range = (97.0, 99.0)
    valence_splits = []
    site_definitions = [
        {"name": "T1", "capacity": 6.0, "priority": ["Si{4+}", "Al{3+}"]},
        {"name": "T2", "capacity": 3.0, "priority": ["Al{3+}", "Ti{4+}"]},
        {
            "name": "B",
            "capacity": 2.0,
            "priority": ["Fe{2+}", "Mg{2+}", "Mn{2+}"],
        },
        {
            "name": "A",
            "capacity": 1.0,
            "priority": ["Na{+}", "K{+}", "Ca{2+}"],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Cordierite end-member proportions (%).

        H₂O-Crd from A-site occupancy, Mg-Crd/Fe-Crd/Mn-Crd from B-site
        R²⁺ fractions (excluding A-site component).
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        fe = self._col(elem_apfu, "Fe{2+}")
        mg = self._col(elem_apfu, "Mg{2+}")
        mn = self._col(elem_apfu, "Mn{2+}")
        na = self._col(elem_apfu, "Na{+}")
        k = self._col(elem_apfu, "K{+}")
        ca = self._col(elem_apfu, "Ca{2+}")

        r2 = fe + mg + mn
        r2_safe = r2.replace(0, 1)
        x_fe = (fe / r2_safe).where(r2 > 0, 0.0)
        x_mg = (mg / r2_safe).where(r2 > 0, 0.0)
        x_mn = (mn / r2_safe).where(r2 > 0, 0.0)

        a_total = (na + k + ca).clip(upper=1.0)
        b_frac = 1.0 - a_total

        result = pd.DataFrame(index=idx)
        result["H₂O-Crd"] = (a_total * 100).round(3)
        result["Mg-Crd"] = (x_mg * b_frac * 100).round(3)
        result["Fe-Crd"] = (x_fe * b_frac * 100).round(3)
        result["Mn-Crd"] = (x_mn * b_frac * 100).round(3)
        return result


Crd = Cordierite()


# ---------------------------------------------------------------------------
# Ilmenite — (Fe²⁺,Mg,Mn)TiO₃  (3 oxygens, 2 cations)
# ---------------------------------------------------------------------------


class Ilmenite(Mineral):
    """Ilmenite group — 3 oxygens, 2 cations.

    Sites: A(1.0), B(1.0).
    Fe³⁺ estimated via Droop (1987).

    End-members: Ilm (FeTiO₃), Gk (MgTiO₃), Pph (MnTiO₃),
    Hem (Fe₂O₃), Chr (FeCr₂O₄).
    """

    name = "Ilmenite"
    n_oxygens = 3
    ideal_cations = 2
    analytical_total_range = (93.0, 100.5)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {
            "name": "A",
            "capacity": 1.0,
            "priority": ["Fe{2+}", "Mg{2+}", "Mn{2+}", "Fe{3+}"],
        },
        {
            "name": "B",
            "capacity": 1.0,
            "priority": ["Ti{4+}", "Fe{3+}", "Al{3+}", "Cr{3+}"],
        },
    ]

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Ilmenite end-member proportions from site-allocated APFU (%).

        Uses structural formula site allocations so fractions are based on
        capacity-clipped site occupancies. Product of A-site and B-site
        fractions for each end-member.
        """
        sf = self.site_allocations(df, units)
        idx = sf.index

        def _site_ion(site: str, ion: str) -> pd.Series:
            key = (site, ion)
            return sf[key] if key in sf.columns else pd.Series(0.0, index=idx)

        # A-site fractions
        fe2_a = _site_ion("A", "Fe{2+}")
        fe3_a = _site_ion("A", "Fe{3+}")
        mg_a = _site_ion("A", "Mg{2+}")
        mn_a = _site_ion("A", "Mn{2+}")
        a_total = fe2_a + fe3_a + mg_a + mn_a
        a_safe = a_total.replace(0, 1)

        x_fe2_a = (fe2_a / a_safe).where(a_total > 0, 0.0)
        x_mg_a = (mg_a / a_safe).where(a_total > 0, 0.0)
        x_mn_a = (mn_a / a_safe).where(a_total > 0, 0.0)
        x_fe3_a = (fe3_a / a_safe).where(a_total > 0, 0.0)

        # B-site fractions
        ti_b = _site_ion("B", "Ti{4+}")
        fe3_b = _site_ion("B", "Fe{3+}")
        al_b = _site_ion("B", "Al{3+}")
        cr_b = _site_ion("B", "Cr{3+}")
        b_total = ti_b + fe3_b + al_b + cr_b
        b_safe = b_total.replace(0, 1)

        x_ti_b = (ti_b / b_safe).where(b_total > 0, 0.0)
        x_fe3_b = (fe3_b / b_safe).where(b_total > 0, 0.0)
        x_cr_b = (cr_b / b_safe).where(b_total > 0, 0.0)

        # A × B products
        ilm = x_fe2_a * x_ti_b
        gk = x_mg_a * x_ti_b
        pph = x_mn_a * x_ti_b
        hem = x_fe3_a * x_fe3_b
        chr_ = x_fe3_a * x_cr_b

        total = ilm + gk + pph + hem + chr_
        total_safe = total.replace(0, 1)

        result = pd.DataFrame(index=idx)
        result["Ilm"] = (ilm / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Gk"] = (gk / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Pph"] = (pph / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Hem"] = (hem / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Chr"] = (chr_ / total_safe * 100).where(total > 0, 0.0).round(3)
        return result


Ilm = Ilmenite()


# ---------------------------------------------------------------------------
# Spinel — AB₂O₄  (4 oxygens, 3 cations)
# ---------------------------------------------------------------------------


class Spinel(Mineral):
    """Spinel group — 4 oxygens, 3 cations.

    Sites: T(1.0), M(2.0).  Inverse spinel — T can host divalent
    cations, M hosts trivalent cations.
    Fe³⁺ estimated via Droop (1987).

    End-members (product of site fractions, %):
    Spl (MgAl₂O₄), Herc (FeAl₂O₄), Chrm (FeCr₂O₄),
    Mtc (Fe₃O₄), Gahn (ZnAl₂O₄), Frank (Fe₂TiO₄),
    Jac (MnAl₂O₄), Ulv (Mg₂TiO₄), Spss (Mn₃O₄).
    """

    name = "Spinel"
    n_oxygens = 4
    ideal_cations = 3
    analytical_total_range = (93.0, 100.5)
    valence_splits = [{"element": "Fe", "method": "droop"}]
    site_definitions = [
        {
            "name": "T",
            "capacity": 1.0,
            "priority": [
                "Mg{2+}",
                "Fe{2+}",
                "Zn{2+}",
                "Mn{2+}",
                "Fe{3+}",
                "Al{3+}",
                "Cr{3+}",
                "Ti{4+}",
            ],
        },
        {
            "name": "M",
            "capacity": 2.0,
            "priority": [
                "Al{3+}",
                "Cr{3+}",
                "Fe{3+}",
                "Ti{4+}",
                "Mg{2+}",
                "Fe{2+}",
                "Mn{2+}",
            ],
        },
    ]

    def _preprocess_oxides(self, oxide_df: pd.DataFrame) -> pd.DataFrame:
        """Merge Fe₂O₃ into FeO before Droop Fe³⁺ split.

        Spinel analyses often report both FeO and Fe₂O₃. Merge Fe₂O₃
        into FeO before normalisation, then apply Droop Fe³⁺ split.
        """
        return _calc.fe2o3_to_feo(oxide_df)

    def end_members(self, df: pd.DataFrame, units: str = "wt%") -> pd.DataFrame:
        """Spinel end-member proportions via product of site fractions (%).

        End-members are products of T-site and M-site cation fractions,
        reflecting the inverse spinel disorder model.
        """
        elem_apfu = self._raw_apfu(df, units)
        idx = elem_apfu.index

        mg = self._col(elem_apfu, "Mg{2+}")
        fe2 = self._col(elem_apfu, "Fe{2+}")
        zn = self._col(elem_apfu, "Zn{2+}")
        mn = self._col(elem_apfu, "Mn{2+}")
        fe3 = self._col(elem_apfu, "Fe{3+}")
        al = self._col(elem_apfu, "Al{3+}")
        cr = self._col(elem_apfu, "Cr{3+}")
        ti = self._col(elem_apfu, "Ti{4+}")

        # T-site fractions
        t_total = mg + fe2 + zn + mn + fe3 + al + cr + ti
        t_safe = t_total.replace(0, 1)
        x_mg_t = (mg / t_safe).where(t_total > 0, 0.0)
        x_fe2_t = (fe2 / t_safe).where(t_total > 0, 0.0)
        x_zn_t = (zn / t_safe).where(t_total > 0, 0.0)
        x_mn_t = (mn / t_safe).where(t_total > 0, 0.0)

        # M-site fractions (2 sites)
        m_total = al + cr + fe3 + ti + mg + fe2 + mn
        m_safe = m_total.replace(0, 1)
        x_mg_m = (mg / m_safe).where(m_total > 0, 0.0)
        x_al_m = (al / m_safe).where(m_total > 0, 0.0)
        x_cr_m = (cr / m_safe).where(m_total > 0, 0.0)
        x_fe3_m = (fe3 / m_safe).where(m_total > 0, 0.0)
        x_ti_m = (ti / m_safe).where(m_total > 0, 0.0)
        x_fe2_m = (fe2 / m_safe).where(m_total > 0, 0.0)
        x_mn_m = (mn / m_safe).where(m_total > 0, 0.0)

        # Product-of-site-fractions end-members
        spl = x_mg_t * x_al_m**2
        herc = x_fe2_t * x_al_m**2
        chrm = x_fe2_t * x_cr_m**2
        mtc = x_fe2_t * x_fe3_m**2
        gahn = x_zn_t * x_al_m**2
        frank = x_fe2_t * x_ti_m * x_fe2_m
        jac = x_mn_t * x_al_m**2
        ulv = x_mg_t * x_ti_m * x_mg_m
        spss = x_mn_t * x_mn_m**2

        # Compute total to normalize
        total = spl + herc + chrm + mtc + gahn + frank + jac + ulv + spss
        total_safe = total.replace(0, 1)

        # Normalized percentages (may not sum to 100 due to minor components)
        # Report as fractions of the computed total
        result = pd.DataFrame(index=idx)
        result["Spl"] = (spl / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Herc"] = (herc / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Chrm"] = (chrm / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Mtc"] = (mtc / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Gahn"] = (gahn / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Frank"] = (frank / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Jac"] = (jac / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Ulv"] = (ulv / total_safe * 100).where(total > 0, 0.0).round(3)
        result["Spss"] = (spss / total_safe * 100).where(total > 0, 0.0).round(3)
        return result


Spl = Spinel()
