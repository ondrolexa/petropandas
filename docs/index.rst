petropandas
===========

A pandas-accessor library for processing electron microprobe (EMPA) mineral
analyses — convert oxide wt% to APFU, compute structural formulas, estimate
Fe\ :sup:`3+`/Fe\ :sup:`2+`, calculate end-members, validate stoichiometry,
and produce publication-ready plots.

Highlights
----------

- **Seamless pandas integration** — all operations available as DataFrame
  accessors (``df.oxides``, ``df.moles``, ``df.apfu``, ``df.mineral``,
  ``df.bulk``)
- **16 IMA mineral objects** — Garnet, Feldspar, Pyroxenes, Micas, Chlorite,
  Epidote, Amphibole, and more
- **16 THERMOCALC-compatible minerals** — polynomial end-member calculations
  from ``tc-mp51.txt``
- **Fe\ :sup:`3+`/Fe\ :sup:`2+` estimation** — Droop (1987) and Schumacher
  (1991) methods
- **Bulk composition tools** — CIPW norm, THERMOCALC/PerpleX/MAGEMin output
  formatting
- **Publication-ready plots** — ScatterPlot, TernaryPlot, ProfilePlot with
  grouped data support

Quick Example
-------------

.. code-block:: python

   from petropandas import pd, Grt

   df = pd.read_csv("analyses.csv")

   # End-member proportions for garnet
   df.mineral.end_members(Grt)

   # Plot garnet compositions on a ternary diagram
   from petropandas import TernaryPlot

   t = TernaryPlot(top="Prp+Sps", left="Alm", right="Grs")
   t.add(df.mineral.end_members(Grt))
   t.show()

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   tutorial
   api/index
