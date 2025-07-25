import numpy as np
import pandas as pd
from periodictable.formulas import formula

oxides = {
    "Al": "Al2O3",
    "Si": "SiO2",
    "Na": "Na2O",
    "K": "K2O",
    "Ca": "CaO",
    "Mg": "MgO",
    "Mn": "MnO",
    "Fe": "FeO",
    "Ti": "TiO2",
    "P": "P2O5",
    "Cr": "Cr2O3",
    "Y": "Y2O3",
}


def formula2wt(s):
    f = formula(s)
    s = pd.Series()
    for atom, n in f.atoms.items():
        if atom.symbol in oxides:
            ox = formula(oxides[atom.symbol])
            s[str(ox)] = n * ox.mass / ox.atoms[atom]
    return s.to_frame().T


class MineralNotCalculated(Exception):
    def __init__(self):
        super().__init__("Structural formula not yet calculated")


class Site:
    """Class to store mineral site information

    Args:
        name (str): name of site
        ncat (int): ideal number of cations
        candidates (list): list of cations in population order

    Attributes:
        name (str): name of site
        ncat (int): ideal number of cations
        candidates (list): list of cations in population order
        atoms (dict): populated atoms
        free (float): remaining space in site, i.e. `Ideal - Occupied`
    """

    def __init__(self, name, ncat, candidates):
        self.name = name
        self.ncat = ncat
        self.candidates = candidates
        self.atoms = {}

    def __repr__(self):
        return f"Site {self.name}[{self.ncat}][{sum(self.atoms.values())}]"

    def add(self, atom, amount, force=False):
        """Add atom to site

        Args:
            atom (str): cation name
            amount (float): number of attoms to add
            force (bool): If ``False`` atoms are added up to ideal occupancy. When
                ``True`` full amount is added. Default ``False``

        """
        if force:
            if atom in self.atoms:
                self.atoms[atom] += amount
            else:
                self.atoms[atom] = amount
        else:
            free = self.free
            if free > 0:
                if free > amount:
                    if atom in self.atoms:
                        self.atoms[atom] += amount
                    else:
                        self.atoms[atom] = amount
                else:
                    self.atoms[atom] = free

    def get(self, atom, fraction=False):
        """Get number of given atom on site

        Args:
            atom (str): atom name

        """
        if fraction:
            return self.atoms.get(atom, 0) / self.ncat
        else:
            return self.atoms.get(atom, 0)

    @property
    def free(self):
        return self.ncat - sum(self.atoms.values())


class StrucForm:
    """Class to manipulate mineral structural formula

    Args:
        mineral (Mineral): mineral instance

    Attributes:
        mineral (Mineral): mineral instance
        sites (list): list of sites
        reminder (pandas.Series): remainder after site population or empty
    """

    def __init__(self, mineral):
        self.mineral = mineral
        self.sites = [Site(*s) for s in self.mineral.structure]
        self.reminder = pd.Series()

    def __repr__(self):
        if not self.reminder.empty:
            res = ""
            for site in sorted(self.sites, key=lambda site: site.name):
                s = ""
                for atom, val in site.atoms.items():
                    s += f"{atom}({val:.3f})"
                res += f"[{s}]"
            return res
        else:
            return "Not yet calculated"

    def get(self, atom) -> np.floating:
        """Get total number of given atom in structure

        Args:
            atom (str): atom name

        """
        return np.sum([s.get(atom) for s in self.sites])

    def site(self, name) -> Site | None:
        """Get site by name

        Args:
            name (str): site name

        """
        sites = [s for s in self.sites if s.name == name]
        if sites:
            return sites[0]

    @property
    def apfu(self) -> pd.Series:
        """Calculate mineral atom p.f.u based on site occupancies

        Returns:
            pandas.Series: atoms p.f.u
        """
        if not self.reminder.empty:
            apfu = pd.Series(
                [self.get(e) for e in self.reminder.index], index=self.reminder.index
            )
            # Add atoms not in analysis
            for site in self.sites:
                for atom in site.candidates:
                    if atom not in apfu:
                        apfu[atom] = 0.0
            return apfu
        else:
            raise MineralNotCalculated()

    @property
    def check_stechiometry(self) -> np.floating:
        """Calculate average missfit of populated and ideal cations on sites"""
        if not self.reminder.empty:
            return np.mean([abs(s.free) / s.ncat for s in self.sites])
        else:
            raise MineralNotCalculated()


class Mineral:
    """Base Mineral class to store mineral structure and endmembers calculation

    Attributes:
        noxy (int): number of oxygens
        ncat (int): number of cations
        needsFe (str): `"Fe2"` for only FeO oxide needed, `"Fe3"` for both FeO and
            Fe2O3 oxide needed, `None` for no Fe needed
        structure (tuple, optional): structural formula definition. Iterable collection
            of tuples `(name, ncat, list of cations)` e.g.
            `("T", 4, ["Si{4+}", "Al{3+}"])`. Tuples are ordered in population order.
        has_structure (bool): True when structure is defined
        has_endmembers (bool): True when endmembers method is defined

    """

    def __init__(self):
        # Subclasses should define this!
        self.structure = ()
        self.noxy = 0
        self.needsFe = False

    def __repr__(self):
        return type(self).__name__

    @property
    def ncat(self):
        return sum([s[1] for s in self.structure])

    @property
    def has_endmembers(self):
        return hasattr(self, "endmembers")

    @property
    def has_structure(self):
        return self.structure != ()

    def endmembers(self, cations, force=False):
        raise NotImplementedError("Subclasses should implement this!")

    def calculate(self, cations, force=False) -> StrucForm:
        """Calculate structural formula from cations p.f.u

        Args:
            cations (pandas.Series): cations p.f.u

        Returns:
            StrucForm: structural formula

        """
        sf = StrucForm(self)
        lastsite = {}
        for site in sf.sites:
            for atom in site.candidates:
                available = cations.get(atom, 0.0) - sf.get(atom)
                if available > 0:
                    site.add(atom, available)
                    lastsite[atom] = site

        # Force site occupancy
        if force:
            occ = pd.Series([sf.get(e) for e in cations.index], index=cations.index)
            reminder = cations - occ
            for atom, r in reminder.items():
                if (r > 0) & (atom in lastsite):
                    lastsite[atom].add(atom, r, force=True)
        # reminder
        occ = pd.Series([sf.get(e) for e in cations.index], index=cations.index)
        sf.reminder = cations - occ
        return sf

    def apfu(self, cations, force=False) -> pd.Series:
        """Calculate mineral atom p.f.u based on site occupancies

        Args:
            cations (pandas.Series): cations p.f.u

        Returns:
            pandas.Series: atoms p.f.u
        """
        sf = self.calculate(cations, force=force)
        return sf.apfu

    def check_stechiometry(self, cations, force=False) -> np.floating:
        """Calculate average missfit of populated and ideal cations on sites

        Args:
            cations (pandas.Series): cations p.f.u

        Returns:
            pandas.Series: average missfit
        """
        sf = self.calculate(cations, force=force)
        return sf.check_stechiometry


class Garnet_Fe2(Mineral):
    """Garnet using total Fe with 4 endmembers"""

    def __init__(self):
        super().__init__()
        self.noxy = 12
        self.needsFe = "Fe2"
        # fmt: off
        self.structure = (
            ("Z", 3, ["Si{4+}", "Al{3+}"]),
            ("Y", 2, ["Si{4+}", "Al{3+}", "Ti{4+}", "Cr{3+}", "Mg{2+}", "Fe{2+}", "Mn{2+}"]),
            ("X", 3, ["Y{3+}", "Mg{2+}", "Fe{2+}", "Mn{2+}", "Ca{2+}", "Na{+}"]),
        )
        # fmt: on

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        esum = apfu["Fe{2+}"] + apfu["Mn{2+}"] + apfu["Mg{2+}"] + apfu["Ca{2+}"]
        em = {
            "Alm": apfu["Fe{2+}"] / esum,
            "Prp": apfu["Mg{2+}"] / esum,
            "Sps": apfu["Mn{2+}"] / esum,
            "Grs": apfu["Ca{2+}"] / esum,
        }
        return pd.Series(em)


class Garnet(Mineral):
    """Garnet using Fe2 and Fe3 with 6 endmembers"""

    def __init__(self):
        super().__init__()
        self.noxy = 12
        self.needsFe = "Fe3"
        # fmt: off
        self.structure = (
            ("Z", 3, ["Si{4+}", "Al{3+}", "Fe{3+}"]),
            ("Y", 2, ["Si{4+}", "Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{3+}", "Mg{2+}", "Fe{2+}", "Mn{2+}"]),
            ("X", 3, ["Y{3+}", "Mg{2+}", "Fe{2+}", "Mn{2+}", "Ca{2+}", "Na{+}"]),
        )
        # fmt: on

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        s8t = apfu["Fe{2+}"] + apfu["Mn{2+}"] + apfu["Mg{2+}"] + apfu["Ca{2+}"]
        s6t = apfu["Ti{4+}"] + apfu["Al{3+}"] + apfu["Cr{3+}"] + apfu["Fe{3+}"]
        # end members in mol%
        em = {
            "Alm": apfu["Fe{2+}"] / s8t,
            "Prp": apfu["Mg{2+}"] / s8t,
            "Sps": apfu["Mn{2+}"] / s8t,
            "Grs": (apfu["Al{3+}"] / s6t) * (apfu["Ca{2+}"] / s8t),
            "Adr": (apfu["Fe{3+}"] / s6t) * (apfu["Ca{2+}"] / s8t),
            "Uv": (apfu["Cr{3+}"] / s6t) * (apfu["Ca{2+}"] / s8t),
            "CaTi": (apfu["Ti{4+}"] / s6t) * (apfu["Ca{2+}"] / s8t),
        }
        return pd.Series(em)


class Feldspar(Mineral):
    """Feldspar with 3 endmembers"""

    def __init__(self):
        super().__init__()
        self.noxy = 8
        self.needsFe = None
        self.structure = (
            ("T", 4, ["Si{4+}", "Al{3+}"]),
            ("A", 1, ["K{+}", "Na{+}", "Ca{2+}"]),
        )

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        alk = apfu["Ca{2+}"] + apfu["Na{+}"] + apfu["K{+}"]
        em = {
            "An": apfu["Ca{2+}"] / alk,
            "Ab": apfu["Na{+}"] / alk,
            "Or": apfu["K{+}"] / alk,
        }
        return pd.Series(em)


class Pyroxene_Fe2(Mineral):
    """Pyroxen using total Fe with 3 endmembers"""

    def __init__(self):
        super().__init__()
        self.noxy = 6
        self.needsFe = "Fe2"
        self.structure = (
            ("T", 2, ["Si{4+}", "Al{3+}"]),
            ("M1", 1, ["Al{3+}", "Ti{4+}", "Cr{3+}", "Mn{2+}", "Mg{2+}", "Fe{2+}"]),
            ("M2", 1, ["Mg{2+}", "Fe{2+}", "Ca{2+}", "Na{+}", "K{+}"]),
        )

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        esum = apfu["Fe{2+}"] + apfu["Mn{2+}"] + apfu["Mg{2+}"] + apfu["Ca{2+}"]
        em = {
            "En": apfu["Mg{2+}"] / esum,
            "Wo": apfu["Ca{2+}"] / esum,
            "Fs": (apfu["Fe{2+}"] + apfu["Mn{2+}"]) / esum,
        }
        return pd.Series(em)


class Pyroxene(Mineral):
    """Pyroxene with Na-Cr with 6 endmembers"""

    def __init__(self):
        super().__init__()
        self.noxy = 6
        self.needsFe = "Fe3"
        # fmt: off
        self.structure = (
            ("T", 2, ["Si{4+}", "Al{3+}", "Fe{3+}"]),
            ("M1", 1, ["Al{3+}", "Ti{4+}", "Fe{3+}", "Cr{3+}", "Mn{2+}", "Mg{2+}", "Fe{2+}"]),
            ("M2", 1, ["Mg{2+}", "Fe{2+}", "Ca{2+}", "Na{+}", "K{+}"]),
        )
        # fmt: on

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        A = np.array(
            [
                apfu["Ca{2+}"],
                apfu["Al{3+}"],
                apfu["Fe{3+}"],
                apfu["Mg{2+}"],
                apfu["Fe{2+}"],
                apfu["Cr{3+}"],
            ]
        )
        M = np.array(
            [
                [0, 0, 0, 2, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        em = A @ np.linalg.inv(M)
        em = em / em.sum()
        return pd.Series(em, index=["En", "Wo", "Fs", "Jd", "Aeg", "Kos"])


class DiMica(Mineral):
    """DiOct Micas"""

    def __init__(self):
        super().__init__()
        self.noxy = 11
        self.needsFe = "Fe2"
        self.structure = (
            ("T", 4, ["Si{4+}", "Al{3+}"]),
            ("M", 2, ["Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{2+}", "Mn{2+}", "Mg{2+}"]),
            ("I", 1, ["Ba{2+}", "Ca{2+}", "Na{+}", "K{+}"]),
        )

    def endmembers(self, cations, force=False):
        sf = self.calculate(cations, force=force)
        apfu = self.apfu(cations, force=force)
        Xm = sf.site("M").get("Al{3+}") - 1
        XCel = 1 - Xm  # total amout (Fe Al-Celadonite + Mg Al-Celadonite)
        XMg = apfu["Mg{2+}"] / (apfu["Mg{2+}"] + apfu["Fe{2+}"])
        Isum = apfu["Ca{2+}"] + apfu["Na{+}"] + apfu["K{+}"]
        XMPM = (
            Isum * Xm
        )  # Sum of Ca + Na + K multiplied by the total fraction of Ms, Pg, Mrg, and Prl
        em = {
            "Al-Celadonite": XMg * XCel,
            "Fe Al-Celadonite": XCel - XMg * XCel,
            "Pyrophyllite": Xm - XMPM,
            "Margarite": XMPM * apfu["Ca{2+}"] / Isum,
            "Paragonite": XMPM * apfu["Na{+}"] / Isum,
            "Muscovite": XMPM * apfu["K{+}"] / Isum,
        }
        return pd.Series(em)
