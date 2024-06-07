# petropandas

[![Release](https://img.shields.io/github/v/release/ondrolexa/petropandas)](https://img.shields.io/github/v/release/ondrolexa/petropandas)
[![Build status](https://img.shields.io/github/actions/workflow/status/ondrolexa/petropandas/main.yml?branch=main)](https://github.com/ondrolexa/petropandas/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ondrolexa/petropandas/branch/main/graph/badge.svg)](https://codecov.io/gh/ondrolexa/petropandas)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ondrolexa/petropandas)](https://img.shields.io/github/commit-activity/m/ondrolexa/petropandas)
[![License](https://img.shields.io/github/license/ondrolexa/petropandas)](https://img.shields.io/github/license/ondrolexa/petropandas)

Pandas accessors for petrologists

- **Github repository**: <https://github.com/ondrolexa/petropandas/>
- **Documentation** <https://ondrolexa.github.io/petropandas/>

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

## GitHub actions

The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

## Documentation

To finalize the set-up for publishing to PyPi or Artifactory, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).

## Releasing a new version

- Create an API Token on [Pypi](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/ondrolexa/petropandas/settings/secrets/actions/new).
- Create a [new release](https://github.com/ondrolexa/petropandas/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
