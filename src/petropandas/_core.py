"""Oxide and ion utilities built on periodictable."""

from __future__ import annotations

from functools import lru_cache

from periodictable import O as _O
from periodictable import formula


# ---------------------------------------------------------------------------
# EMPA column aliases -> standard oxide formula
# ---------------------------------------------------------------------------

ALIASES: dict[str, str] = {
    "FeO*": "FeO",
    "FeOT": "FeO",
    "FeO tot": "FeO",
    "FeOt": "FeO",
    "FeO Total": "FeO",
    "FeO(T)": "FeO",
    "Fe2O3*": "Fe2O3",
    "Fe2O3T": "Fe2O3",
    "Fe2O3 tot": "Fe2O3",
    "Fe2O3t": "Fe2O3",
    "Fe2O3 Total": "Fe2O3",
    "Fe2O3(T)": "Fe2O3",
    "H2O_PLUS": "H2O",
    "H2O+": "H2O",
    "H2OPLUS": "H2O",
    "H2OP": "H2O",
}


# ---------------------------------------------------------------------------
# Oxide helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _is_oxide(col: str) -> bool:
    """Return True if *col* parses as a formula containing oxygen."""
    try:
        return _O in formula(col).atoms
    except Exception:
        return False


def _oxide_cols(df) -> list[str]:
    """Return columns of *df* that are parseable oxides, preserving order."""
    return [c for c in df.columns if _is_oxide(c)]


@lru_cache(maxsize=None)
def _element_of(oxide: str) -> str:
    """Return the cation symbol for an oxide formula.

    Args:
        oxide: Oxide formula string (e.g. ``"Fe2O3"``).

    Returns:
        Element symbol (e.g. ``"Fe"``).
    """
    atoms = formula(oxide).atoms
    for el, count in atoms.items():
        if el != _O:
            return el.symbol
    return ""


@lru_cache(maxsize=None)
def _cations_per(oxide: str) -> int:
    """Return the number of cation atoms in one formula unit.

    Args:
        oxide: Oxide formula string (e.g. ``"Al2O3"``).

    Returns:
        Number of cations (e.g. ``2`` for Al₂O₃).
    """
    atoms = formula(oxide).atoms
    return sum(count for el, count in atoms.items() if el != _O)


@lru_cache(maxsize=None)
def _oxygens_per(oxide: str) -> int:
    """Return the number of oxygen atoms in one formula unit.

    Args:
        oxide: Oxide formula string (e.g. ``"Al2O3"``).

    Returns:
        Number of oxygens (e.g. ``3`` for Al₂O₃).
    """
    return formula(oxide).atoms.get(_O, 0)


@lru_cache(maxsize=None)
def MW(oxide: str) -> float:
    """Return molecular weight via periodictable.

    Args:
        oxide: Oxide formula string (e.g. ``"SiO2"``).

    Returns:
        Molecular weight in g/mol.
    """
    return formula(oxide).mass


# ---------------------------------------------------------------------------
# Ion helpers
# ---------------------------------------------------------------------------


def _ion_name(element_symbol: str, charge: int) -> str:
    """Format an ion column name.

    Args:
        element_symbol: Element symbol (e.g. ``"Fe"``).
        charge: Ionic charge (e.g. ``2``).

    Returns:
        Ion string (e.g. ``"Fe{2+}"``).  Charge of +/-1 omits the number
        (``"Na{+}"``, ``"Cl{-}"``).
    """
    sign = "+" if charge > 0 else "-"
    if abs(charge) == 1:
        return f"{element_symbol}{{{sign}}}"
    return f"{element_symbol}{{{abs(charge)}{sign}}}"


def _parse_ion(col: str) -> tuple[str, int] | None:
    """Parse an ion column name into (element, charge).

    Args:
        col: Column name (e.g. ``"Fe{2+}"``).

    Returns:
        Tuple of (element symbol, charge), or None if not an ion.
    """
    try:
        f = formula(col)
        if f.charge != 0:
            for atom, count in f.atoms.items():
                if hasattr(atom, "charge") and count == 1:
                    return atom.symbol, atom.charge
        return None
    except Exception:
        return None


def _ion_to_oxide(element_symbol: str, charge: int) -> str:
    """Map an element and charge to the standard EMPA oxide formula.

    Args:
        element_symbol: Element symbol (e.g. ``"Fe"``).
        charge: Ionic charge (e.g. ``3``).

    Returns:
        Oxide formula (e.g. ``"Fe2O3"``).
    """
    if charge % 2 == 0:
        n_o = charge // 2
        return f"{element_symbol}O{n_o}" if n_o > 1 else f"{element_symbol}O"
    n_o = charge
    return f"{element_symbol}2O" if n_o == 1 else f"{element_symbol}2O{n_o}"


def _element_symbol_from_ion(col: str) -> str | None:
    """Extract the element symbol from an ion or oxide column name.

    Args:
        col: Column name (e.g. ``"Fe{2+}"`` or ``"FeO"``).

    Returns:
        Element symbol, or None if unparseable.
    """
    result = _parse_ion(col)
    if result is not None:
        return result[0]
    try:
        return _element_of(col)
    except Exception:
        return None


_ELEMENT_CHARGE: dict[str, int] = {
    "Si": 4,
    "Ti": 4,
    "Al": 3,
    "Cr": 3,
    "Fe": 2,
    "Mn": 2,
    "Mg": 2,
    "Ca": 2,
    "Na": 1,
    "K": 1,
    "Ba": 2,
    "Sr": 2,
    "Zn": 2,
    "P": 5,
    "V": 3,
}


def _element_charge(element_symbol: str) -> int:
    """Return the common oxidation state for an EMPA-reported element.

    Args:
        element_symbol: Element symbol (e.g. ``"Fe"``).

    Returns:
        Default charge (e.g. ``2`` for Fe as Fe²⁺ in FeO).
    """
    return _ELEMENT_CHARGE.get(element_symbol, 2)


def _detect_col(df, element: str) -> str:
    """Find the first column in *df* containing *element*.

    Works for both oxide names (``'FeO'``) and ion names (``'Fe{2+}'``).

    Args:
        df: DataFrame to search.
        element: Element symbol to find.

    Returns:
        Matching column name.

    Raises:
        KeyError: If no column contains the element.
    """
    for col in df.columns:
        try:
            atoms = formula(col).atoms
            if any(atom.symbol == element for atom in atoms):
                return col
        except Exception:
            continue
    raise KeyError(f"No column found for element {element!r}")


def _detect_cols(df, element: str) -> list[str]:
    """Find all columns in *df* containing *element*.

    Args:
        df: DataFrame to search.
        element: Element symbol to find.

    Returns:
        List of matching column names.
    """
    result = []
    for col in df.columns:
        try:
            atoms = formula(col).atoms
            if any(atom.symbol == element for atom in atoms):
                result.append(col)
        except Exception:
            continue
    return result
