# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2026-09-04

### Added
- `df.cations.total_charge()`: total positive charge per row, summed over ion-named columns
- `df.<accessor>.calc(new_col, expr)`, available on all `_BaseAccessor` subclasses: add a column computed from a `pandas.eval()`-style expression, reusing the plotting axes' expression syntax (`petropandas._calc.eval_expr`)

## [0.2.1] - 2026-08-29

### Added
- `sum(*, groupby=None)` available on all accessors alongside `mean()`
- `df.mineral.stoichiometry_quality(mineral)`: single 0–1 score, the mean of `check_stoichiometry()`'s `cation_deviation`, `site_vacancies`, and `leftover_cations`
- `Mineral.__str__` returns `.abbreviation` and `Mineral.__repr__` returns `.name`; every built-in mineral now sets `.abbreviation` matching its instance name (e.g. `Grt.abbreviation == "Grt"`)
- `mdb` mineral registry (`from petropandas import mdb`): `all()`, `by_name()`, `by_abbreviation()` (case-insensitive), and `names`/`abbreviations` properties over the 16 built-in minerals

## [0.2] - 2026-08-29

### Added
- mineral-fractionation mass balance for spherical grains
- `mean()`, `reframe()`, `normalize()`, and `select()` available on all accessors

### Changed
- `normalized()` renamed to `normalize(to=100.0)`, with a configurable target row sum
- `mean()`'s `weights` argument now also accepts an array-like of numbers
- added a `tests` optional-dependency group; `dev` now pulls in `lab`/`tests`/`docs` plus `ruff`/`pre-commit`
- CI now measures test coverage and uploads it to Codecov; release workflow pins Python to 3.12

## [0.1.4] - 2026-07-13

### Added
- dataframe kwarg added to petrosoftware bulk calculations
- petroplots.profile autogrouping and autoscaling implemented

## [0.1.3] - 2025-12-16

### Added

- profile accepts ax, show and high
- mbe added to MAGEMin
- Garnet_TC added

## [0.1.2] - 2025-11-27

### Added

- Ternary accepts pandas expressions as args or `c`, `s`, `v` kwargs
- Ternary contour added
- Client for postresql petrodb database API
- accessors plotting methods plot, heatmap and boxplot Added
- drop columns with only `NA` by default

### Fixed

- pyroxene recoded
- molprop method available for all accessors
- oxides order could be defined in config

## [0.1.1] - 2025-10-27

- Initial release
