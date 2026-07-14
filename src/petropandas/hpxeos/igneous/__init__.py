"""Phase set for the igneous axfile (tc-ig51NCKFMASHTOCr.txt, NCKFMASHTOCr system).

Green, Holland, Powell, Weller & Riel (2025) - a corrigendum to Holland, Green &
Powell (2018) - covering subalkaline magmatic systems from peridotites through to
granites. Despite the different model generation from `hpxeos.metapelite`/
`hpxeos.metabasite`, several phase blocks turn out to be numerically identical (or an
exact Mn-free subset of) blocks already implemented there - those are reused directly
rather than duplicated. See each name's source module for which case it is.
"""

from .biotite import TC_bi_G25, Biotite
from .clinopyroxene import TC_cpx_W24, Clinopyroxene
from .cordierite import TC_cd_G25, Cordierite
from .garnet import TC_g_W24, Garnet
from .ilmenite import TC_ilm_W24, Ilmenite
from .olivine import TC_ol_H18, Olivine
from .orthopyroxene import TC_opx_W24, Orthopyroxene
from .spinel import TC_spl_T21, Spinel
from ..metabasite import TC_hb, Amphibole
from ..metapelite import TC_ep, TC_mu, TC_pl4tr, Epidote, Muscovite, Plagioclase

__all__ = [
    "Epidote",
    "Muscovite",
    "Plagioclase",
    "Amphibole",
    "Cordierite",
    "Olivine",
    "Ilmenite",
    "Garnet",
    "Biotite",
    "Spinel",
    "Orthopyroxene",
    "Clinopyroxene",
    "TC_ep",
    "TC_mu",
    "TC_pl4tr",
    "TC_hb",
    "TC_cd_G25",
    "TC_ol_H18",
    "TC_ilm_W24",
    "TC_g_W24",
    "TC_bi_G25",
    "TC_spl_T21",
    "TC_opx_W24",
    "TC_cpx_W24",
]
