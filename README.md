# petropandas

A pandas-accessor library for processing electron microprobe (EMPA) mineral analyses — convert oxide wt% to APFU, compute structural formulas, estimate Fe³⁺/Fe²⁺, calculate end-members, validate stoichiometry, and produce publication-ready plots.

## Install

```bash
uv sync
```

### Optional dependencies

```bash
uv sync --extra lab
```

## Quick start

```python
from petropandas import pd, Grt
from petropandas.data import minerals as df
# minerals is a DataFrame of example analyses
```

Filter to oxide columns:

```python
df.oxides()
```

Re-order columns in standard petrological order (SiO₂ first, volatiles last):

```python
df.oxides.sorted()
```

Select rows by searching for "Garnet" in column "Mineral":

```python
g = df.oxides.select("Garnet", on="Mineral")
```

Compute element APFU with Fe³⁺/Fe²⁺ splitting for garnet (12 oxygens):

```python
g.mineral.apfu(Grt)
```

Site-allocated structural formula:

```python
g.mineral.site_allocations(Grt)
```

End-member proportions:

```python
g.mineral.end_members(Grt)
```

Validate analysis quality against ideal stoichiometry:

```python
g.mineral.check_stoichiometry(Grt)
```

Returns a DataFrame of 0–1 scores per criterion (1 = perfect fit).

## Supported minerals

| Mineral | Instance | O₂ | Cations | Fe method | End-members |
|---------|----------|----|---------|-----------|-------------|
| Garnet | `Grt` | 12 | 8 | Droop | Prp, Alm, Sps |
| Garnet (Fe³⁺) | `GrtFe3` | 12 | 8 | Matrix inversion | Prp, Alm, Sps, Grs, Adr, Uvr |
| Feldspar | `Fsp` | 8 | 5 | — | An, Ab, Or |
| Clinopyroxene | `Cpx` | 6 | 4 | Droop | Jd, Ae, Wo, En, Fs |
| Orthopyroxene | `Opx` | 6 | 4 | Droop | MgTs, Wo, En, Fs |
| Muscovite | `Ms` | 11 | 7 | — | Al-Cel, Fe-Al-Cel, Prl, Mrg, Pg, Ms, Trioct |
| Biotite | `Bt` | 11 | 7 | — | Phl, Ann, Eas, Sid, Dioct |
| Staurolite | `St` | 48 | — | — | Fe-St, Mg-St, Zn-St, Mn-St |
| Chlorite | `Chl` | 14* | — | — | Clin, Cham, Mg-Sud, Fe-Sud |
| Epidote | `Ep` | 12.5 | 8 | FeO→Fe₂O₃ | Czo, Ep, Pmn, Muk, Taw |
| Amphibole | `Amp` | 23 | 15 | Schumacher | Tr, Act, Ed, F-Ed, Prg, F-Prg, Tsch, Rct, Win, Glau, F-Glau, Rieb, Mg-Rieb |
| Titanite | `Ttn` | 5 | 3 | FeO→Fe₂O₃ | Ttn, Al-Ttn, Fe-Ttn, Mal, Other |
| Chloritoid | `Cld` | 12 | 8 | Droop | Cld, Mgcld, Mncld |
| Cordierite | `Crd` | 18 | 11 | — | H₂O-Crd, Mg-Crd, Fe-Crd, Mn-Crd |
| Ilmenite | `Ilm` | 3 | 2 | Droop | Ilm, Gk, Pph, Hem, Chr |
| Spinel | `Spl` | 4 | 3 | Droop | Spl, Herc, Chrm, Mtc, Gahn, Frank, Jac, Ulv, Spss |

\* Chlorite uses 28-charge normalization (n_oxygens=14 effective).

THERMOCALC activity-composition (a-x) solution models are available through the
`petropandas.hpxeos` subpackage, covering three real THERMOCALC axfiles —
`hpxeos.metapelite`, `hpxeos.metabasite`, `hpxeos.igneous` — each exposing ready-to-use
`TC_<abbreviation>` instances (e.g. `TC_g`, `TC_pl4tr`) alongside their `Phase` classes:

```python
from petropandas.hpxeos.metapelite import TC_g

df.mineral.apfu(TC_g)
df.mineral.site_allocations(TC_g)
df.mineral.end_members(TC_g)
df.mineral.check_stoichiometry(TC_g)
```

`Phase` subclasses with order-disorder variables (e.g. Biotite `Q`, Augite `Qfm`/`Qal`)
accept an optional `order_parameters` dict, passed through
`df.mineral.end_members(mineral, order_parameters={...})`.

## API reference

### OxidesAccessor (`df.oxides`)

| Method | Description |
|--------|-------------|
| `df.oxides()` | Return copy with only recognised oxide columns (wt%) |
| `df.oxides.sorted()` | Return oxide wt% with columns in petrological order |
| `df.oxides.normalized()` | Normalise wt% to sum to 100% |
| `df.oxides.mean(*, groupby=None)` | Mean oxide wt%, optionally grouped by a column |
| `df.oxides.split_valence(elem, method, n_oxy, ideal_cat)` | Split element into low/high charge oxides (wt%) |
| `df.oxides.oxidize(o_excess)` | Split FeO into FeO/Fe₂O₃ by excess oxygen (THERMOCALC) |
| `df.oxides.reduce()` | Merge Fe₂O₃ back into FeO equivalent |
| `df.oxides.apatite_correction()` | Remove CaO bound in apatite, zero P₂O₅ |

### MolesAccessor (`df.moles`)

| Method | Description |
|--------|-------------|
| `df.moles()` | Return oxide columns as molar proportions |
| `df.moles.normalized()` | Normalise molar proportions to sum to 100% |

### ApfuAccessor (`df.apfu`)

| Method | Description |
|--------|-------------|
| `df.apfu(n_oxygens=N)` | Atoms per formula unit (oxygen basis) |
| `df.apfu(n_cations=N)` | Atoms per formula unit (cation basis) |

All callable accessors auto-convert from the current `petro_units` attr.
Chains work seamlessly: `df.oxides().moles().oxides()` roundtrips back to wt%.
`df.apfu(n_oxygens=N).oxides()` converts APFU back to oxide wt% (ratios preserved).

### MineralAccessor (`df.mineral`)

| Method | Description |
|--------|-------------|
| `df.mineral.apfu(mineral)` | Element APFU with valence splits for a mineral |
| `df.mineral.site_allocations(mineral)` | Site allocations with hierarchical (site, cation) columns |
| `df.mineral.end_members(mineral)` | End-member proportions (%) |
| `df.mineral.check_stoichiometry(mineral)` | Stoichiometry validation scores (0–1) |

### BulkAccessor (`df.bulk`)

| Method | Description |
|--------|-------------|
| `df.bulk()` | Return cleaned copy in wt% |
| `df.bulk.mean(*, groupby=None, weights=None)` | Mean oxide wt%; optional weighted mean |
| `df.bulk.cipw()` | Simple CIPW normative mineralogy |
| `df.bulk.alumina_saturation(classify=False)` | A/NK and A/CNK molar ratios; optional Shand classification |
| `df.bulk.oxide_ratios()` | Common ratios (Mg#, FeOT, total alkalis, K/Na, etc.) |
| `df.bulk.TCbulk(*, system, ...)` | THERMOCALC bulk script output |
| `df.bulk.Perplexbulk(*, system, ...)` | PerpleX thermodynamic component list output |
| `df.bulk.MAGEMin(*, db, ...)` | MAGEMin bulk input file output |

### Series accessor (`series.mineral`)

| Method | Description |
|--------|-------------|
| `series.mineral.is_oxide` | True if column is a recognised oxide |
| `series.mineral.element` | Element symbol |
| `series.mineral.molecular_weight` | Molecular weight |
| `series.mineral.to_mole()` | Convert one column to moles |
| `series.mineral.to_cation(n_oxygens, total_oxygens)` | Convert one column to APFU |

## Plotting

Three plot classes are available for visualising microprobe and end-member data:

```python
from petropandas import ScatterPlot, TernaryPlot, ProfilePlot
```

### ScatterPlot

X/y scatter plot where axes are `pandas.eval()` expressions over DataFrame columns.

```python
s = ScatterPlot("Prp", "Sps+Grs")
s.add(garnet_df, label="Garnet 1")
s.add(other_df, label="Garnet 2")
fig, ax = s.render()
```

### TernaryPlot

Ternary diagram built on plain matplotlib (no mpltern dependency). Axes are `pandas.eval()` expressions.

```python
s = TernaryPlot("Prp", "Sps", "Grs")
s.add(garnet_df, label="Garnet 1")
fig, ax = s.render()
```

### ProfilePlot

Line plot of DataFrame columns against their index, with optional dual y-axes for columns spanning different value scales.

```python
s = ProfilePlot(secondary_columns=["Alm"])
s.add(profile_df, label="Profile 1")
fig, ax = s.render()
```

All three inherit from `BasePlot` and share `add()`, `render()`, `show()`, and `savefig()` methods.

## Stoichiometry checking

`df.mineral.check_stoichiometry(mineral)` returns a DataFrame scored 0–1 for each criterion:

| Criterion | What it checks |
|-----------|----------------|
| `analytical_total` | Oxide wt% sum vs mineral-specific ideal range |
| `cation_deviation` | Total APFU vs ideal cation count |
| `charge_balance` | Total positive charge vs expected from oxygen count |
| `fe3+_validity` | Fe³⁺ and Fe²⁺ non-negative after valence splitting |
| `site_vacancies` | Mean site occupancy fraction |
| `leftover_cations` | Fraction of APFU not assigned to any site |
| `tetrahedral_fill` | T-site sum vs T-site capacity |

Inapplicable criteria are dropped (all-NaN columns removed).

## Fe³⁺/Fe²⁺ estimation methods

- **Droop (1987)**: `method="droop"` — charge-balance approach on total cations
- **Schumacher (1991)**: `method="schumacher"` — oxygen-excess approach
- **FeO→Fe₂O₃ conversion**: Epidote and Titanite internally convert all FeO to Fe₂O₃ before normalisation
- **Spinel**: Merges Fe₂O₃ into FeO before Droop split (inverse spinel model)
- **`df.oxides.split_valence()`**: General-purpose valence splitting on any element
- **`df.oxides.oxidize(o_excess)`**: THERMOCALC excess-oxygen convention (Fe₂O₃ = 2 × o_excess)

## Conventions

- Ion column names use periodictable notation: `Fe{2+}`, `Fe{3+}`, `Si{4+}`, `Na{+}`
- Structural formula columns: `(site, cation)` tuples (e.g. `("Z", "Si{4+}")`, `("X", "Fe{2+}")`)
- Unit tracking via `df.attrs["petro_units"]`: `"wt%"` (default) → `"moles"` → `"apfu"`
- Callable accessors (`df.oxides()`, `df.moles()`, `df.apfu()`) auto-convert from current units
- `Mineral` instances are configuration objects — stateless, reusable
- Internal modules are underscore-prefixed (`_calc.py`, `_core.py`, `_minerals.py`, `_plotting.py`)
