# Getting Started

## Installation

petropandas requires Python 3.12 or later.

```bash
uv add petropandas
```

Or with pip:

```bash
pip install petropandas
```

### Optional Dependencies

```bash
# For Jupyter notebook workflows
uv add petropandas --optional lab

# For building this documentation
uv add petropandas --optional docs
```

## Basic Usage

### Loading Data

```python
from petropandas import pd
from petropandas.data import minerals, pyroxenes, grt_profile, bulk

# Built-in example datasets
df = minerals  # 16 mineral analyses
px = pyroxenes  # 5 pyroxene spot analyses
gp = grt_profile  # 99-point garnet profile
b = bulk  # 9 bulk rock compositions

# Or load your own CSV
df = pd.read_csv("analyses.csv")
```

petropandas automatically detects oxide columns (e.g. `SiO2`, `TiO2`, `Al2O3`, `FeO`, `Fe2O3`, `MnO`, `MgO`, `CaO`, `Na2O`, `K2O`, `P2O5`) and standardizes column names via built-in aliases.

### Working with Oxides

```python
# View oxide columns in standard petrological order
df.oxides.sorted()

# Normalize to 100 wt%
df.oxides.normalized()

# Group averages
df.oxides.mean(groupby="rock_type")

# Split FeO into FeO and Fe2O3 using THERMOCALC convention
df.oxides.oxidize(o_excess=0.01)
```

### Unit Conversions

```python
# Convert to molar proportions
df.moles()

# Convert to atoms per formula unit (APFU)
df.apfu(n_oxygens=12)

# Unified conversion interface
import petropandas._calc as _calc
_calc.convert(df, to_unit="apfu", n_oxygens=12)
```

### Mineral Calculations

```python
from petropandas import Grt, Cpx, Amp

# APFU for garnet (12 oxygens)
df.mineral.apfu(Grt)

# End-member proportions
df.mineral.end_members(Grt)

# Site allocations
df.mineral.site_allocations(Grt)

# Stoichiometry quality check (0-1 scores)
df.mineral.check_stoichiometry(Grt)
```

### Bulk Composition Analysis

```python
# Weighted mean of bulk compositions
b.bulk.mean(weights="SiO2")

# CIPW normative mineralogy
b.bulk.cipw()

# Alumina saturation indices
b.bulk.alumina_saturation()

# THERMOCALC bulk composition output
b.bulk.TCbulk(system="MnNCKFMASHTO")
```

### Plotting

```python
from petropandas import ScatterPlot, TernaryPlot, ProfilePlot

# Scatter plot
sp = ScatterPlot("MgO", "FeO")
sp.add(df, label="samples")
sp.show()

# Ternary diagram
tp = TernaryPlot(top="Prp", left="Alm", right="Grs")
tp.add(df.mineral.end_members(Grt))
tp.show()

# Compositional profile
pp = ProfilePlot()
pp.add(spot_data)
pp.show()
```
