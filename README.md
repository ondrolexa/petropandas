# petropandas

[![Release](https://img.shields.io/github/v/release/ondrolexa/petropandas)](https://img.shields.io/github/v/release/ondrolexa/petropandas)
[![Build status](https://img.shields.io/github/actions/workflow/status/ondrolexa/petropandas/main.yml?branch=main)](https://github.com/ondrolexa/petropandas/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ondrolexa/petropandas)](https://img.shields.io/github/commit-activity/m/ondrolexa/petropandas)
[![License](https://img.shields.io/github/license/ondrolexa/petropandas)](https://img.shields.io/github/license/ondrolexa/petropandas)

Pandas accessors for petrologists

## Getting started

First, import the petropandas. Note that numpy, matplotlib.pyplot, pandas and seaborn are also
imported using common aliases np, plt, pd and sns:

```python
from petropandas import *
```

You are now ready to use petropandas tools.

```python
df = pd.read_excel("some/folder/data.xlsx")
df.oxides.molprop()
df.oxides.cations(noxy=12)

df.ree.normalize(reservoir='CI Chondrites', reference='McDonough & Sun 1995')
```
