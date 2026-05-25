# petropandas

[![Release](https://img.shields.io/github/v/release/ondrolexa/petropandas)](https://img.shields.io/github/v/release/ondrolexa/petropandas)
[![Build status](https://img.shields.io/github/actions/workflow/status/ondrolexa/petropandas/testing.yml?branch=main)](https://github.com/ondrolexa/petropandas/actions/workflows/testing.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ondrolexa/petropandas)](https://img.shields.io/github/commit-activity/m/ondrolexa/petropandas)
[![License](https://img.shields.io/github/license/ondrolexa/petropandas)](https://img.shields.io/github/license/ondrolexa/petropandas)

Pandas accessors for petrologists

## Installation

```bash
pip install petropandas
```

### Optional extras

| Extra | Includes | Description |
|---|---|---|
| `database` | `requests` | Client for the remote petrodb REST API (`petropandas.database`) |
| `extra` | `jupyterlab` | JupyterLab support |
| `tests` | `pytest`, `nbval` | Run the test suite |
| `docs` | `mkdocs`, ... | Build documentation locally |
| `dev` | all of the above | Development setup |

```bash
pip install petropandas[database]      # basic + database client
pip install petropandas[dev]           # full development setup
```

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

### Database client (optional)

The `petropandas.database` module provides a client for a remote petrodb REST API.
Install the extra and use it alongside the accessors:

```bash
pip install petropandas[database]
```

```python
from petropandas.database import PetroDB

db = PetroDB('http://127.0.0.1:8000', 'user', 'password')
project = db.projects(name="MyProject")
sample = project.samples(name="DB250")
df = sample.spots.df(mineral="Grt")
```
