# petropandas

[![Release](https://img.shields.io/github/v/release/ondrolexa/petropandas)](https://img.shields.io/github/v/release/ondrolexa/petropandas)
[![Build status](https://img.shields.io/github/actions/workflow/status/ondrolexa/petropandas/testing.yml?branch=main)](https://github.com/ondrolexa/petropandas/actions/workflows/testing.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ondrolexa/petropandas)](https://img.shields.io/github/commit-activity/m/ondrolexa/petropandas)
[![License](https://img.shields.io/github/license/ondrolexa/petropandas)](https://img.shields.io/github/license/ondrolexa/petropandas)

Pandas accessors for petrologists

## Getting started

`petropandas` provides several `pandas.DataFrame` accessors to seemlesly integrate
common petrological calculations to your Python data analysis workflow.

```python
from petropandas import pd, mindb
```

For more details check the [documentation](https://petropandas.readthedocs.io/).

```python
df = pd.read_excel("some/folder/data.xlsx")
df.oxides.molprop()
df.oxides.cations(noxy=12)

df.ree.normalize(reservoir='CI Chondrites', reference='McDonough & Sun 1995')
```
