"""Phase set for the metabasite axfile (tc-mb51NCKFMASHTO.txt, NCKFMASHTO system).

Several of this file's phase blocks are numerically identical (or an exact Mn-free
subset of) blocks already implemented for the metapelite axfile
(tc-mp51MnNCKFMASHTO.txt, `hpxeos.metapelite`) - those are reused directly rather than
duplicated. See each name's source module for which case it is.
"""

from ..metapelite import (
    Epidote,
    Ilmenite,
    KFeldspar,
    KFeldsparCbar1,
    Plagioclase,
    PlagioclaseCbar1,
    Spinel,
    TC_ep,
    TC_ilm,
    TC_k4tr,
    TC_ksp,
    TC_pl4tr,
    TC_plc,
    TC_sp,
)
from .amphibole import Amphibole, TC_hb
from .augite import Augite, TC_aug
from .biotite import Biotite, TC_bi
from .chlorite import Chlorite, TC_chl
from .garnet import Garnet, TC_g
from .ilmenite_mixed import IlmeniteMixed, TC_ilmm
from .muscovite import Muscovite, TC_mu
from .olivine import Olivine, TC_ol
from .omphacite import Omphacite, TC_dio
from .orthopyroxene import Orthopyroxene, TC_opx
from .peristerite import Peristerite, TC_abc
from .plagioclase_ibar1 import PlagioclaseIbar1, TC_pli

__all__ = [
    "Amphibole",
    "Augite",
    "Biotite",
    "Chlorite",
    "Epidote",
    "Garnet",
    "Ilmenite",
    "IlmeniteMixed",
    "KFeldspar",
    "KFeldsparCbar1",
    "Muscovite",
    "Olivine",
    "Omphacite",
    "Orthopyroxene",
    "Peristerite",
    "Plagioclase",
    "PlagioclaseCbar1",
    "PlagioclaseIbar1",
    "Spinel",
    "TC_abc",
    "TC_aug",
    "TC_bi",
    "TC_chl",
    "TC_dio",
    "TC_ep",
    "TC_g",
    "TC_hb",
    "TC_ilm",
    "TC_ilmm",
    "TC_k4tr",
    "TC_ksp",
    "TC_mu",
    "TC_ol",
    "TC_opx",
    "TC_pl4tr",
    "TC_plc",
    "TC_pli",
    "TC_sp",
]
