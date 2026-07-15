Minerals
========

All mineral classes and instances are importable from the top-level package.

Base Class
----------

.. autoclass:: petropandas._minerals.Mineral
   :members:
   :undoc-members:
   :show-inheritance:

IMA Mineral Instances
---------------------

These are pre-configured :class:`~petropandas._minerals.Mineral` instances
using IMA-standard abbreviations.

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Instance
     - O\ :sub:`2`
     - Cations
     - Fe method
   * - ``Grt``
     - 12
     - 8
     - Droop
   * - ``GrtFe3``
     - 12
     - 8
     - Matrix inversion
   * - ``Fsp``
     - 8
     - 5
     - —
   * - ``Cpx``
     - 6
     - 4
     - Droop
   * - ``Opx``
     - 6
     - 4
     - Droop
   * - ``Ms``
     - 11
     - 7
     - —
   * - ``Bt``
     - 11
     - 7
     - —
   * - ``St``
     - 48
     - —
     - —
   * - ``Chl``
     - 14\*
     - —
     - —
   * - ``Ep``
     - 12.5
     - 8
     - FeO→Fe\ :sub:`2`\ O\ :sub:`3`
   * - ``Amp``
     - 23
     - 15
     - Schumacher
   * - ``Ttn``
     - 5
     - 3
     - FeO→Fe\ :sub:`2`\ O\ :sub:`3`
   * - ``Cld``
     - 12
     - 8
     - Droop
   * - ``Crd``
     - 18
     - 11
     - —
   * - ``Ilm``
     - 3
     - 2
     - Droop
   * - ``Spl``
     - 4
     - 3
     - Droop

\* Chlorite uses 28-charge normalization (``n_oxygens=14`` effective).

THERMOCALC a-x Models (``hpxeos``)
-----------------------------------

THERMOCALC activity-composition (a-x) solution models live in the
``petropandas.hpxeos`` subpackage, transcribed directly from real THERMOCALC
axfiles. Three phase sets are available — ``petropandas.hpxeos.metapelite``,
``petropandas.hpxeos.metabasite``, and ``petropandas.hpxeos.igneous`` — each
built on the shared :class:`~petropandas.hpxeos.base.Phase` base class (itself
a :class:`~petropandas._minerals.Mineral` subclass), and each exposing
pre-configured ``TC_<abbreviation>`` instances ready for ``df.mineral.*``:

.. code-block:: python

   from petropandas.hpxeos.metapelite import TC_g

   df.mineral.apfu(TC_g)
   df.mineral.end_members(TC_g)

Phases with order-disorder variables not recoverable from bulk composition
(e.g. Biotite's ``Q``, Augite's ``Qfm``/``Qal``) accept an
``order_parameters`` dict, forwarded via
``df.mineral.end_members(mineral, order_parameters={...})``.

.. autoclass:: petropandas.hpxeos.base.Phase
   :members:
   :undoc-members:
   :show-inheritance:

Phase Sets
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 30 25 10

   * - Subpackage
     - Axfile
     - System
     - Phases
   * - ``hpxeos.metapelite``
     - ``tc-mp51MnNCKFMASHTO.txt``
     - MnNCKFMASHTO
     - 19
   * - ``hpxeos.metabasite``
     - ``tc-mb51NCKFMASHTO.txt``
     - NCKFMASHTO
     - 19
   * - ``hpxeos.igneous``
     - ``tc-ig51NCKFMASHTOCr.txt``
     - NCKFMASHTOCr
     - 12

hpxeos.metapelite (MnNCKFMASHTO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 12 25 15

   * - Class
     - Singleton
     - Mineral
     - End-members
   * - ``Biotite``
     - ``TC_bi``
     - Biotite
     - 7
   * - ``Chlorite``
     - ``TC_chl``
     - Chlorite
     - 8
   * - ``Chloritoid``
     - ``TC_ctd``
     - Chloritoid
     - 4
   * - ``Cordierite``
     - ``TC_cd``
     - Cordierite
     - 4
   * - ``Epidote``
     - ``TC_ep``
     - Epidote
     - 3
   * - ``Garnet``
     - ``TC_g``
     - Garnet
     - 5
   * - ``Ilmenite``
     - ``TC_ilm``
     - Ilmenite
     - 3
   * - ``IlmeniteMixed``
     - ``TC_ilmm``
     - Ilmenite-hematite
     - 5
   * - ``KFeldspar``
     - ``TC_k4tr``
     - K-feldspar (4TR)
     - 3
   * - ``KFeldsparCbar1``
     - ``TC_ksp``
     - K-feldspar (Cbar1 ASF)
     - 3
   * - ``Magnetite``
     - ``TC_mt1``
     - Magnetite
     - 3
   * - ``Margarite``
     - ``TC_ma``
     - Margarite
     - 6
   * - ``Muscovite``
     - ``TC_mu``
     - Muscovite
     - 6
   * - ``Orthopyroxene``
     - ``TC_opx``
     - Orthopyroxene
     - 7
   * - ``Plagioclase``
     - ``TC_pl4tr``
     - Plagioclase (4TR)
     - 3
   * - ``PlagioclaseCbar1``
     - ``TC_plc``
     - Plagioclase (Cbar1 ASF)
     - 3
   * - ``Sapphirine``
     - ``TC_sa``
     - Sapphirine
     - 5
   * - ``Spinel``
     - ``TC_sp``
     - Spinel
     - 4
   * - ``Staurolite``
     - ``TC_st``
     - Staurolite
     - 5

hpxeos.metabasite (NCKFMASHTO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 12 28 15 18

   * - Class
     - Singleton
     - Mineral
     - End-members
     - Note
   * - ``Amphibole``
     - ``TC_hb``
     - Clinoamphibole
     - 11
     - new
   * - ``Augite``
     - ``TC_aug``
     - Augite (Ca-Mg-Fe cpx)
     - 8
     - new
   * - ``Biotite``
     - ``TC_bi``
     - Biotite
     - 6
     - Mn-free subset
   * - ``Chlorite``
     - ``TC_chl``
     - Chlorite
     - 7
     - Mn-free subset
   * - ``Garnet``
     - ``TC_g``
     - Garnet
     - 4
     - Mn-free subset
   * - ``IlmeniteMixed``
     - ``TC_ilmm``
     - Ilmenite-hematite
     - 4
     - Mn-free subset
   * - ``Muscovite``
     - ``TC_mu``
     - Muscovite
     - 6
     - new (Ca: ``mam``)
   * - ``Olivine``
     - ``TC_ol``
     - Olivine
     - 2
     - new
   * - ``Omphacite``
     - ``TC_dio``
     - Omphacite (Na-Ca cpx)
     - 7
     - new
   * - ``Orthopyroxene``
     - ``TC_opx``
     - Orthopyroxene
     - 6
     - Mn-free subset
   * - ``Peristerite``
     - ``TC_abc``
     - Low-albite peristerite
     - 2
     - new
   * - ``PlagioclaseIbar1``
     - ``TC_pli``
     - Plagioclase (Ibar1 ASF)
     - 3
     - new
   * - ``Plagioclase``
     - ``TC_pl4tr``
     - Plagioclase (4TR)
     - 3
     - re-exported
   * - ``KFeldspar``
     - ``TC_k4tr``
     - K-feldspar (4TR)
     - 3
     - re-exported
   * - ``KFeldsparCbar1``
     - ``TC_ksp``
     - K-feldspar (Cbar1 ASF)
     - 3
     - re-exported
   * - ``PlagioclaseCbar1``
     - ``TC_plc``
     - Plagioclase (Cbar1 ASF)
     - 3
     - re-exported
   * - ``Epidote``
     - ``TC_ep``
     - Epidote
     - 3
     - re-exported
   * - ``Spinel``
     - ``TC_sp``
     - Spinel
     - 4
     - re-exported
   * - ``Ilmenite``
     - ``TC_ilm``
     - Ilmenite
     - 3
     - re-exported

hpxeos.igneous (NCKFMASHTOCr)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 16 28 15 18

   * - Class
     - Singleton
     - Mineral
     - End-members
     - Note
   * - ``Biotite``
     - ``TC_bi_G25``
     - Biotite
     - 6
     - Mn-free subset
   * - ``Clinopyroxene``
     - ``TC_cpx_W24``
     - Clinopyroxene (Cr/Ti/K)
     - 10
     - new
   * - ``Cordierite``
     - ``TC_cd_G25``
     - Cordierite
     - 3
     - Mn-free subset
   * - ``Garnet``
     - ``TC_g_W24``
     - Garnet (Cr/Ti-bearing)
     - 6
     - new
   * - ``Ilmenite``
     - ``TC_ilm_W24``
     - Ilmenite
     - 5
     - new
   * - ``Olivine``
     - ``TC_ol_H18``
     - Olivine (incl. monticellite)
     - 4
     - new
   * - ``Orthopyroxene``
     - ``TC_opx_W24``
     - Orthopyroxene (Cr/Ti/Na)
     - 9
     - new
   * - ``Spinel``
     - ``TC_spl_T21``
     - Spinel
     - 8
     - new
   * - ``Epidote``
     - ``TC_ep``
     - Epidote
     - 3
     - re-exported
   * - ``Muscovite``
     - ``TC_mu``
     - Muscovite
     - 6
     - re-exported
   * - ``Plagioclase``
     - ``TC_pl4tr``
     - Plagioclase (4TR)
     - 3
     - re-exported
   * - ``Amphibole``
     - ``TC_hb``
     - Clinoamphibole
     - 11
     - re-exported
