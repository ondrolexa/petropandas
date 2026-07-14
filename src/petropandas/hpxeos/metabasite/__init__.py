"""Phase set for the metabasite axfile (tc-mb51NCKFMASHTO.txt, NCKFMASHTO system).

Several of this file's phase blocks are numerically identical (or an exact Mn-free
subset of) blocks already implemented for the metapelite axfile
(tc-mp51MnNCKFMASHTO.txt, `hpxeos.metapelite`) - those are reused directly rather than
duplicated. See each name's source module for which case it is.
"""

from .amphibole import TC_hb, Amphibole
from .augite import TC_aug, Augite
from .biotite import TC_bi, Biotite
from .chlorite import TC_chl, Chlorite
from .garnet import TC_g, Garnet
from .ilmenite_mixed import TC_ilmm, IlmeniteMixed
from .muscovite import TC_mu, Muscovite
from .olivine import TC_ol, Olivine
from .omphacite import TC_dio, Omphacite
from .orthopyroxene import TC_opx, Orthopyroxene
from .peristerite import TC_abc, Peristerite
from .plagioclase_ibar1 import TC_pli, PlagioclaseIbar1
from ..metapelite import (
    TC_ep,
    TC_ilm,
    TC_k4tr,
    TC_ksp,
    TC_pl4tr,
    TC_plc,
    TC_sp,
    Epidote,
    Ilmenite,
    KFeldspar,
    KFeldsparCbar1,
    Plagioclase,
    PlagioclaseCbar1,
    Spinel,
)

__all__ = [
    "Plagioclase",
    "KFeldspar",
    "KFeldsparCbar1",
    "PlagioclaseCbar1",
    "Epidote",
    "Spinel",
    "Ilmenite",
    "Muscovite",
    "Garnet",
    "Orthopyroxene",
    "Biotite",
    "Chlorite",
    "IlmeniteMixed",
    "Olivine",
    "Peristerite",
    "PlagioclaseIbar1",
    "Augite",
    "Omphacite",
    "Amphibole",
    "TC_ep",
    "TC_ilm",
    "TC_k4tr",
    "TC_ksp",
    "TC_pl4tr",
    "TC_plc",
    "TC_sp",
    "TC_mu",
    "TC_g",
    "TC_opx",
    "TC_bi",
    "TC_chl",
    "TC_ilmm",
    "TC_ol",
    "TC_abc",
    "TC_pli",
    "TC_aug",
    "TC_dio",
    "TC_hb",
]
