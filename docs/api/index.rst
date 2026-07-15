API Reference
=============

petropandas extends pandas DataFrames with mineral analysis functionality
through registered accessors.

DataFlow Accessors
------------------

These accessors convert between different representations of oxide data.

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Accessor
     - Input Units
     - Output
     - Description
   * - ``df.oxides()``
     - any
     - wt%
     - Return oxide columns in weight percent
   * - ``df.moles()``
     - any
     - moles
     - Return molar proportions
   * - ``df.apfu(n_oxygens=N)``
     - any
     - APFU
     - Atoms per formula unit with ion-named columns

All accessors auto-convert from the current unit tracked in
``df.attrs["petro_units"]`` (default: ``"wt%"``).

Mineral Analysis Accessor
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Accessor
     - Description
   * - ``df.mineral.apfu(mineral)``
     - Compute APFU with Fe\ :sup:`3+`/Fe\ :sup:`2+` splitting
   * - ``df.mineral.site_allocations(mineral)``
     - Allocate cations to crystallographic sites
   * - ``df.mineral.end_members(mineral)``
     - Calculate end-member proportions
   * - ``df.mineral.check_stoichiometry(mineral)``
     - Score analytical quality (0–1)

Bulk Composition Accessor
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Accessor
     - Description
   * - ``df.bulk()``
     - Cleaned copy in wt%
   * - ``df.bulk.mean(*, groupby=None, weights=None)``
     - Mean oxide wt%; optional weighted mean
   * - ``df.bulk.cipw()``
     - CIPW normative mineralogy
   * - ``df.bulk.alumina_saturation()``
     - A/NK and A/CNK molar ratios
   * - ``df.bulk.oxide_ratios()``
     - Mg#, Fe\ :sub:`OT`, total alkalis
   * - ``df.bulk.TCbulk()``
     - THERMOCALC bulk composition
   * - ``df.bulk.Perplexbulk()``
     - PerpleX bulk composition
   * - ``df.bulk.MAGEMin()``
     - MAGEMin bulk composition

Example Data
------------

Built-in example datasets available via ``petropandas.data``:

.. code-block:: python

   from petropandas.data import minerals, grt_profile, bulk, avgpelite

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Dataset
     - Rows
     - Description
   * - ``minerals``
     - 315
     - Analyses across 21 mineral groups (``Mineral`` column)
   * - ``grt_profile``
     - 99
     - Garnet compositional profile
   * - ``bulk``
     - 9
     - Bulk rock compositions
   * - ``avgpelite``
     - 1
     - Average pelite composition

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   accessors
   minerals
   calc
   plotting
   database
