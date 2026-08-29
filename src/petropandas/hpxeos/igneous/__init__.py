"""Phase set for the igneous axfile (tc-ig51NCKFMASHTOCr.txt, NCKFMASHTOCr system).

Green, Holland, Powell, Weller & Riel (2025) - a corrigendum to Holland, Green &
Powell (2018) - covering subalkaline magmatic systems from peridotites through to
granites. Despite the different model generation from `hpxeos.metapelite`/
`hpxeos.metabasite`, several phase blocks turn out to be numerically identical (or an
exact Mn-free subset of) blocks already implemented there - those are reused directly
rather than duplicated. See each name's source module for which case it is.
"""

from ..metabasite import Amphibole, TC_hb
from ..metapelite import Epidote, Muscovite, Plagioclase, TC_ep, TC_mu, TC_pl4tr
from .biotite import Biotite, TC_bi_G25
from .clinopyroxene import Clinopyroxene, TC_cpx_W24
from .cordierite import Cordierite, TC_cd_G25
from .garnet import Garnet, TC_g_W24
from .ilmenite import Ilmenite, TC_ilm_W24
from .olivine import Olivine, TC_ol_H18
from .orthopyroxene import Orthopyroxene, TC_opx_W24
from .spinel import Spinel, TC_spl_T21

__all__ = [
    "Amphibole",
    "Biotite",
    "Clinopyroxene",
    "Cordierite",
    "Epidote",
    "Garnet",
    "Ilmenite",
    "Muscovite",
    "Olivine",
    "Orthopyroxene",
    "Plagioclase",
    "Spinel",
    "TC_bi_G25",
    "TC_cd_G25",
    "TC_cpx_W24",
    "TC_ep",
    "TC_g_W24",
    "TC_hb",
    "TC_ilm_W24",
    "TC_mu",
    "TC_ol_H18",
    "TC_opx_W24",
    "TC_pl4tr",
    "TC_spl_T21",
]
