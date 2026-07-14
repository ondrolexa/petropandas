"""Base class for THERMOCALC a-x solution-model phases.

Each phase block in a THERMOCALC axfile goes through the same three stages:
site fractions -> independent compositional variables -> end-member
proportion polynomials p(...). `Phase` implements the mechanics shared by
every mineral (validating input columns, summing cations onto sites,
orchestrating the pipeline); subclasses supply only the formulas that are
specific to that mineral's axfile block.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd

from petropandas._minerals import Mineral

#: A named order-disorder parameter (e.g. "QAl"), either a single value shared by every
#: analysis or a per-analysis Series.
OrderParameters = dict[str, "float | pd.Series"]


def resolve_order_parameters(
    order_parameters: OrderParameters | None,
    names: Sequence[str],
    index: pd.Index,
    default: float = 0.0,
) -> dict[str, pd.Series]:
    """Fill in `names` from `order_parameters`, broadcasting scalars to `index` and
    defaulting any name not supplied to `default` (fully disordered) rather than
    raising: order parameters describe a mineral's ordering state, which isn't
    recoverable from bulk composition alone, so they are optional caller-supplied
    inputs rather than something derived here."""
    order_parameters = order_parameters or {}
    resolved = {}
    for name in names:
        value = order_parameters.get(name, default)
        resolved[name] = (
            value if isinstance(value, pd.Series) else pd.Series(value, index=index)
        )
    return resolved


class Phase(Mineral, ABC):
    """One THERMOCALC solution-phase a-x model.

    Compositions are DataFrames of cations per formula unit, one row per
    analysis, columns named as `periodictable` ion strings (e.g. "Fe{2+}").

    Subclasses inherit the oxide-wt%-to-APFU conversion, site allocation, and
    stoichiometry-check machinery from :class:`~petropandas._minerals.Mineral`
    (``n_oxygens``, ``ideal_cations``, ``valence_splits``, ``site_definitions``,
    ``analytical_total_range`` must be set as class attributes alongside
    ``sites``/``end_member_names``), so a ``Phase`` instance is usable directly
    through ``df.mineral.apfu/site_allocations/end_members/check_stoichiometry``.
    """

    #: axfile phase abbreviation, e.g. "g" for garnet
    abbreviation: str
    #: {site_name: [cation columns occupying that site]}
    sites: dict[str, list[str]]
    #: end-member names, in the order returned by `proportions`
    end_member_names: list[str]
    #: cation columns that need not be present in `composition` (e.g. an oxidation
    #: state not analyzed); absent values are treated as the site model requires,
    #: typically zero.
    optional_columns: set[str] = set()

    def site_totals(self, composition: pd.DataFrame) -> pd.DataFrame:
        """Sum of cations assigned to each site, per analysis."""
        return pd.DataFrame(
            {
                site: composition[cations].sum(axis=1)
                for site, cations in self.sites.items()
            },
            index=composition.index,
        )

    @abstractmethod
    def site_fractions(self, composition: pd.DataFrame) -> pd.DataFrame:
        """Occupancy of each cation on its site (the axfile's `sf` block), by allocating
        cations from `composition` onto `self.sites` and normalizing per site."""

    @abstractmethod
    def variables(
        self,
        site_fractions: pd.DataFrame,
        order_parameters: OrderParameters | None = None,
    ) -> pd.DataFrame:
        """Independent compositional variables (the axfile's initial-estimate
        block, e.g. x, y, z, m, Q) derived from `site_fractions`. `order_parameters`
        supplies any order-disorder variables that can't be derived from bulk
        composition (see `resolve_order_parameters`)."""

    @abstractmethod
    def end_member_proportions(self, variables: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the p(end-member) polynomials from `variables`."""

    def _validate_columns(self, composition: pd.DataFrame) -> None:
        required = {
            cation for cations in self.sites.values() for cation in cations
        } - self.optional_columns
        missing = required - set(composition.columns)
        if missing:
            raise ValueError(
                f"{type(self).__name__} composition is missing required columns: {sorted(missing)}"
            )

    def proportions(
        self, composition: pd.DataFrame, order_parameters: OrderParameters | None = None
    ) -> pd.DataFrame:
        """End-member molar proportions for each analysis in `composition`."""
        self._validate_columns(composition)
        site_fractions = self.site_fractions(composition)
        variables = self.variables(site_fractions, order_parameters)
        proportions = self.end_member_proportions(variables)[self.end_member_names]
        # skipna=False: a row with NaN proportions (e.g. a genuine 0/0 degenerate
        # composition) must not silently pass just because the non-NaN columns
        # happen to already sum to 1.
        row_sums = proportions.sum(axis=1, skipna=False)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                f"{type(self).__name__} end-member proportions do not sum to 1"
            )
        return proportions

    def end_members(
        self,
        df: pd.DataFrame,
        units: str = "wt%",
        order_parameters: OrderParameters | None = None,
    ) -> pd.DataFrame:
        """Compute end-member proportions as percentages via :meth:`proportions`.

        Overrides :meth:`Mineral.end_members`: rather than the flat
        THERMOCALC-token evaluator, this delegates to the axfile-native
        ``site_fractions`` -> ``variables`` -> ``end_member_proportions``
        pipeline, which already returns proportions summing to 1.

        Args:
            df: DataFrame with oxide columns.
            units: Current units (``"wt%"`` or ``"moles"``).
            order_parameters: Order-disorder variables not recoverable from
                bulk composition (see :func:`resolve_order_parameters`).

        Returns:
            DataFrame of end-member percentages.
        """
        raw = self._raw_apfu(df, units)
        return (self.proportions(raw, order_parameters) * 100).round(3)
