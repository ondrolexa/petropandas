# Overview

petropandas extends pandas DataFrames with mineral analysis functionality through registered accessors.

## DataFlow Accessors

These accessors convert between different representations of oxide data.

| Accessor | Input Units | Output | Description |
|----------|------------|--------|-------------|
| `df.oxides()` | any | wt% | Return oxide columns in weight percent |
| `df.moles()` | any | moles | Return molar proportions |
| `df.apfu(n_oxygens=N)` | any | APFU | Atoms per formula unit with ion-named columns |

All accessors auto-convert from the current unit tracked in `df.attrs["petro_units"]` (default: `"wt%"`).

## Mineral Analysis Accessor

| Accessor | Description |
|----------|-------------|
| `df.mineral.apfu(mineral)` | Compute APFU with Fe³⁺/Fe²⁺ splitting |
| `df.mineral.site_allocations(mineral)` | Allocate cations to crystallographic sites |
| `df.mineral.end_members(mineral)` | Calculate end-member proportions |
| `df.mineral.check_stoichiometry(mineral)` | Score analytical quality (0-1) |

## Bulk Composition Accessor

| Accessor | Description |
|----------|-------------|
| `df.bulk()` | Cleaned copy in wt% |
| `df.bulk.mean(*, groupby=None, weights=None)` | Mean oxide wt%; optional weighted mean |
| `df.bulk.cipw()` | CIPW normative mineralogy |
| `df.bulk.alumina_saturation()` | A/NK and A/CNK molar ratios |
| `df.bulk.oxide_ratios()` | Mg#, FeOT, total alkalis |
| `df.bulk.apatite_correction()` | Remove CaO bound in apatite |
| `df.bulk.TCbulk()` | THERMOCALC bulk composition |
| `df.bulk.Perplexbulk()` | PerpleX bulk composition |
| `df.bulk.MAGEMin()` | MAGEMin bulk composition |

## Example Data

Built-in example datasets available via `petropandas.data`:

```python
from petropandas.data import minerals, pyroxenes, grt_profile, bulk, avgpelite
```

| Dataset | Rows | Description |
|---------|------|-------------|
| `minerals` | 16 | Mixed mineral analyses |
| `pyroxenes` | 5 | Pyroxene spot analyses |
| `grt_profile` | 99 | Garnet compositional profile |
| `bulk` | 9 | Bulk rock compositions |
| `avgpelite` | 1 | Average pelite composition |

## Modules

| Module | Description |
|--------|-------------|
| [Minerals](minerals.md) | Mineral classes and instances |
| [Calculations](calc.md) | Pure calculation functions |
| [Plotting](plotting.md) | Publication-ready plots |
| [Database](database.md) | PetroDB REST client |
