# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- mineral-fractionation mass balance for spherical grains
- `mean()`, `reframe()`, `normalize()`, and `select()` available on all accessors

### Changed
- `normalized()` renamed to `normalize(to=100.0)`, with a configurable target row sum
- `mean()`'s `weights` argument now also accepts an array-like of numbers

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
