# petropandas

A pandas-accessor library for processing electron microprobe (EMPA) mineral analyses — convert oxide wt% to APFU, compute structural formulas, estimate Fe³⁺/Fe²⁺, calculate end-members, validate stoichiometry, and produce publication-ready plots.

## Highlights

- **Seamless pandas integration** — all operations are available as DataFrame accessors (`df.oxides`, `df.moles`, `df.apfu`, `df.mineral`, `df.bulk`)
- **16 IMA mineral objects** — Garnet, Feldspar, Pyroxenes, Micas, Chlorite, Epidote, Amphibole, and more
- **16 THERMOCALC-compatible minerals** — polynomial end-member calculations from `tc-mp51.txt`
- **Fe³⁺/Fe²⁺ estimation** — Droop (1987) and Schumacher (1991) methods
- **Bulk composition tools** — CIPW norm, THERMOCALC/PerpleX/MAGEMin output formatting
- **Publication-ready plots** — ScatterPlot, TernaryPlot, ProfilePlot with grouped data support

## Quick Example

```python
from petropandas import pd, Grt

df = pd.read_csv("analyses.csv")

# End-member proportions for garnet
df.mineral.end_members(Grt)

# Plot garnet compositions on a ternary diagram
from petropandas import TernaryPlot

t = TernaryPlot(top="Prp+Sps", left="Alm", right="Grs")
t.add(df.mineral.end_members(Grt))
t.show()
```

## Links

- [Getting Started](getting-started.md) — Installation and first steps
- [Tutorial](tutorial.md) — Hands-on walkthrough
- [API Reference](api/index.md) — Complete API documentation
