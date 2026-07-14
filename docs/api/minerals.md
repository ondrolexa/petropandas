# Minerals

All mineral classes and instances are importable from the top-level package.

## Base Class

::: petropandas._minerals.Mineral
    options:
      show_source: false
      members_order: source

## IMA Mineral Instances

These are pre-configured `Mineral` instances using IMA-standard abbreviations.

::: petropandas._minerals.Grt

::: petropandas._minerals.Cpx

::: petropandas._minerals.Opx

::: petropandas._minerals.Fsp

::: petropandas._minerals.Ms

::: petropandas._minerals.Bt

::: petropandas._minerals.Amp

::: petropandas._minerals.Chl

::: petropandas._minerals.Cld

::: petropandas._minerals.Crd

::: petropandas._minerals.Ep

::: petropandas._minerals.Ilm

::: petropandas._minerals.Spl

::: petropandas._minerals.St

::: petropandas._minerals.Ttn

::: petropandas._minerals.GrtFe3

## THERMOCALC a-x Models (`hpxeos`)

THERMOCALC activity-composition (a-x) solution models live in the `petropandas.hpxeos`
subpackage, transcribed directly from real THERMOCALC axfiles rather than from
hand-copied polynomial tokens. Three phase sets are available —
`petropandas.hpxeos.metapelite`, `petropandas.hpxeos.metabasite`, and
`petropandas.hpxeos.igneous` — each built on the shared `Phase` base class (itself a
`Mineral` subclass), and each exposing pre-configured `TC_<abbreviation>` instances
ready for `df.mineral.*`:

```python
from petropandas.hpxeos.metapelite import TC_g

df.mineral.apfu(TC_g)
df.mineral.end_members(TC_g)
```

Phases with order-disorder variables not recoverable from bulk composition (e.g.
Biotite's `Q`, Augite's `Qfm`/`Qal`) accept an `order_parameters` dict, forwarded via
`df.mineral.end_members(mineral, order_parameters={...})`.

::: petropandas.hpxeos.base.Phase
    options:
      show_source: false
      members_order: source

## Phase sets

Three subpackages, each built from one real THERMOCALC axfile (checked into the repo
root):

| Subpackage          | Axfile                   | System        | Phases |
|----------------------|---------------------------|----------------|:---:|
| `hpxeos.metapelite`  | `tc-mp51MnNCKFMASHTO.txt` | MnNCKFMASHTO  | 19 |
| `hpxeos.metabasite`  | `tc-mb51NCKFMASHTO.txt`   | NCKFMASHTO    | 19 |
| `hpxeos.igneous`     | `tc-ig51NCKFMASHTOCr.txt` | NCKFMASHTOCr  | 12 |

Phases that are numerically identical across sets (verified by diffing the axfiles'
polynomial blocks, not assumed from matching names) are re-exported directly rather
than duplicated - e.g. `hpxeos.metabasite.Plagioclase is hpxeos.metapelite.Plagioclase`.
Phases that are an exact Mn-free subset of a richer model are implemented as their own
class but documented as such. See each phase module's docstring for which case it is
and how its mass-balance was derived.

### `hpxeos.metapelite` (MnNCKFMASHTO)

| Class              | Singleton | Mineral                  | End-members |
|---------------------|-----------|---------------------------|:---:|
| `Biotite`           | `TC_bi`    | Biotite                  | 7  |
| `Chlorite`          | `TC_chl`   | Chlorite                 | 8  |
| `Chloritoid`        | `TC_ctd`   | Chloritoid               | 4  |
| `Cordierite`        | `TC_cd`    | Cordierite               | 4  |
| `Epidote`           | `TC_ep`    | Epidote                  | 3  |
| `Garnet`            | `TC_g`     | Garnet                   | 5  |
| `Ilmenite`          | `TC_ilm`   | Ilmenite                 | 3  |
| `IlmeniteMixed`     | `TC_ilmm`  | Ilmenite-hematite        | 5  |
| `KFeldspar`         | `TC_k4tr`  | K-feldspar (4TR)         | 3  |
| `KFeldsparCbar1`    | `TC_ksp`   | K-feldspar (Cbar1 ASF)   | 3  |
| `Magnetite`         | `TC_mt1`   | Magnetite                | 3  |
| `Margarite`         | `TC_ma`    | Margarite                | 6  |
| `Muscovite`         | `TC_mu`    | Muscovite                | 6  |
| `Orthopyroxene`     | `TC_opx`   | Orthopyroxene            | 7  |
| `Plagioclase`       | `TC_pl4tr` | Plagioclase (4TR)        | 3  |
| `PlagioclaseCbar1`  | `TC_plc`   | Plagioclase (Cbar1 ASF)  | 3  |
| `Sapphirine`        | `TC_sa`    | Sapphirine               | 5  |
| `Spinel`            | `TC_sp`    | Spinel                   | 4  |
| `Staurolite`        | `TC_st`    | Staurolite               | 5  |

### `hpxeos.metabasite` (NCKFMASHTO)

| Class               | Singleton | Mineral                          | End-members | Note |
|-----------------------|-----------|-----------------------------------|:---:|------|
| `Amphibole`         | `TC_hb`    | Clinoamphibole                   | 11 | new |
| `Augite`            | `TC_aug`   | Augite (Ca-Mg-Fe clinopyroxene)  | 8  | new |
| `Biotite`           | `TC_bi`    | Biotite                          | 6  | Mn-free subset of `metapelite.Biotite` |
| `Chlorite`          | `TC_chl`   | Chlorite                         | 7  | Mn-free subset of `metapelite.Chlorite` |
| `Garnet`            | `TC_g`     | Garnet                           | 4  | Mn-free subset of `metapelite.Garnet` |
| `IlmeniteMixed`     | `TC_ilmm`  | Ilmenite-hematite                | 4  | Mn-free subset of `metapelite.IlmeniteMixed` |
| `Muscovite`         | `TC_mu`    | Muscovite                        | 6  | new (Ca end-member renamed `mam`) |
| `Olivine`           | `TC_ol`    | Olivine                          | 2  | new |
| `Omphacite`         | `TC_dio`   | Omphacite (Na-Ca clinopyroxene)  | 7  | new |
| `Orthopyroxene`     | `TC_opx`   | Orthopyroxene                    | 6  | Mn-free subset of `metapelite.Orthopyroxene` |
| `Peristerite`       | `TC_abc`   | Low-albite peristerite           | 2  | new |
| `PlagioclaseIbar1`  | `TC_pli`   | Plagioclase (Ibar1 ASF)          | 3  | new |
| `Plagioclase`       | `TC_pl4tr` | Plagioclase (4TR)                | 3  | re-exported from `metapelite` |
| `KFeldspar`         | `TC_k4tr`  | K-feldspar (4TR)                 | 3  | re-exported from `metapelite` |
| `KFeldsparCbar1`    | `TC_ksp`   | K-feldspar (Cbar1 ASF)           | 3  | re-exported from `metapelite` |
| `PlagioclaseCbar1`  | `TC_plc`   | Plagioclase (Cbar1 ASF)          | 3  | re-exported from `metapelite` |
| `Epidote`           | `TC_ep`    | Epidote                          | 3  | re-exported from `metapelite` |
| `Spinel`            | `TC_sp`    | Spinel                           | 4  | re-exported from `metapelite` |
| `Ilmenite`          | `TC_ilm`   | Ilmenite                         | 3  | re-exported from `metapelite` |

### `hpxeos.igneous` (NCKFMASHTOCr)

| Class               | Singleton    | Mineral                          | End-members | Note |
|-----------------------|--------------|-----------------------------------|:---:|------|
| `Biotite`           | `TC_bi_G25`  | Biotite                          | 6  | Mn-free subset of `metapelite.Biotite` |
| `Clinopyroxene`     | `TC_cpx_W24` | Clinopyroxene (Cr/Ti/K-bearing)  | 10 | new |
| `Cordierite`        | `TC_cd_G25`  | Cordierite                       | 3  | Mn-free subset of `metapelite.Cordierite` |
| `Garnet`            | `TC_g_W24`   | Garnet (Cr/Ti-bearing)           | 6  | new |
| `Ilmenite`          | `TC_ilm_W24` | Ilmenite                         | 5  | new |
| `Olivine`           | `TC_ol_H18`  | Olivine (incl. monticellite)     | 4  | new |
| `Orthopyroxene`     | `TC_opx_W24` | Orthopyroxene (Cr/Ti/Na-bearing) | 9  | new |
| `Spinel`            | `TC_spl_T21` | Spinel                           | 8  | new |
| `Epidote`           | `TC_ep`      | Epidote                          | 3  | re-exported from `metapelite` |
| `Muscovite`         | `TC_mu`      | Muscovite                        | 6  | re-exported from `metapelite` |
| `Plagioclase`       | `TC_pl4tr`   | Plagioclase (4TR)                | 3  | re-exported from `metapelite` |
| `Amphibole`         | `TC_hb`      | Clinoamphibole                   | 11 | re-exported from `metabasite` |
