Calculations
============

The ``petropandas._calc`` module provides pure functions for unit conversions,
valence splitting, and petrological calculations.

Unit Conversion
---------------

.. autofunction:: petropandas._calc.convert

.. autofunction:: petropandas._calc.to_moles

.. autofunction:: petropandas._calc.to_oxides

.. autofunction:: petropandas._calc.to_apfu

.. autofunction:: petropandas._calc.to_apfu_by_charge

.. autofunction:: petropandas._calc.from_apfu

.. autofunction:: petropandas._calc.molecular_weights

Valence & Oxidation
-------------------

.. autofunction:: petropandas._calc.split_valence

.. autofunction:: petropandas._calc.oxidize_moles

.. autofunction:: petropandas._calc.reduce_moles

.. autofunction:: petropandas._calc.feo_to_fe2o3

.. autofunction:: petropandas._calc.fe2o3_to_feo

Petrological Calculations
-------------------------

.. autofunction:: petropandas._calc.normalize

.. autofunction:: petropandas._calc.alumina_saturation

.. autofunction:: petropandas._calc.oxide_ratios

.. autofunction:: petropandas._calc.apatite_correction

.. autofunction:: petropandas._calc.cipw_norm
