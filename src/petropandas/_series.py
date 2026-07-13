"""Series accessor for individual oxide columns."""

from __future__ import annotations

import pandas as pd

from petropandas._core import MW, _element_of, _is_oxide


@pd.api.extensions.register_series_accessor("mineral")
class MineralSeriesAccessor:
    """Per-column accessor for oxide Series (``series.mineral``)."""

    def __init__(self, obj: pd.Series) -> None:  # noqa: UP007
        self._obj = obj

    @property
    def is_oxide(self) -> bool:
        """True if this Series represents a recognised oxide column."""
        return _is_oxide(str(self._obj.name))

    @property
    def element(self) -> str | None:
        """Element symbol for this oxide, or None if not recognised."""
        try:
            return _element_of(str(self._obj.name))
        except Exception:
            return None

    @property
    def molecular_weight(self) -> float | None:
        """Molecular weight from periodictable, or None if not recognised."""
        try:
            return MW(str(self._obj.name))
        except Exception:
            return None

    def to_mole(self) -> pd.Series:
        """Convert oxide wt% to moles."""
        return self._obj / MW(str(self._obj.name))

    def to_cation(
        self, n_oxygens: int | float, total_oxygens: int | float
    ) -> pd.Series:
        """Convert oxide wt% to APFU for this column.

        Args:
            n_oxygens: Target oxygens per formula unit for the mineral.
            total_oxygens: Sum of oxygen moles for all oxides in the analysis.

        Returns:
            Series of APFU values.
        """
        from petropandas._core import _cations_per

        name = str(self._obj.name)
        return (self._obj / MW(name)) * _cations_per(name) * (n_oxygens / total_oxygens)
