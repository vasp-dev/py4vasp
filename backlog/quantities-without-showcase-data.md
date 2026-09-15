# Quantities that still have random demo data

`py4vasp.demo.calculation` draws most of its data from `_demo/showcase/`, which is
deterministic and physically consistent. The quantities below still come from
`_demo/*.py`, whose unseeded `np.random` is right for the `raw_data` test fixture and
wrong for anything a user looks at. Each group is a separate piece of work.

## The real-space grid family

`density` (and its `tau` and `all_electron` sources), `potential`, `nics`,
`current_density` and `exciton_density` are filled with `wrap_random_data`. This is the
worst of the remaining output: an isosurface of white noise crosses the level at nearly
every voxel, so `plot()` renders confetti rather than a shape. Their examples are
excluded from collection by `_EXAMPLES_NOT_RUNNABLE_YET` in `tests/test_doctest.py`,
and their method-level examples still build a calculation with
`Calculation.from_path(".")`, so they need rewriting as well as new data.

`_demo/showcase/grid.py` is the foundation: a lattice-periodic superposition of
atom-centred Gaussians, summed over the images that reach, cached read-only.
`showcase/density.py` and `showcase/partial_density.py` show the pattern on the
graphite slab. Four things were learned building those and are worth knowing before
starting:

- Every grid quantity of one selection has to share one grid. `bader.BaderAnalysis`
  raises unless the density it builds basins from has the shape of the quantity being
  integrated, and `density`, `potential`, `nics`, `partial_density` and
  `exciton_density` all have a `bader_charge` example.
- Anything evaluated by Fourier transform needs an odd number of grid points. On a cell
  that is not orthogonal an even grid leaves `|G|²` asymmetric under `G → -G` (measured:
  82.2 for the body-centred cell, exactly 0 for an orthogonal one), which makes a
  Poisson solution complex. `grid.grid_for` already returns odd counts.
- The default isolevels are absolute and differ per quantity — density 0.2, potential
  0.0, nics ±1.0, exciton density 0.8 — so the units and widths of the field have to be
  chosen to make them land somewhere meaningful. A user calling `plot()` gets the
  default.
- A density in VASP's convention and an isosurface at an absolute level pull in
  opposite directions. `showcase/density.py` follows VASP (the mean over the grid is
  the electron count) so Bader reports electrons; `showcase/partial_density.py`
  normalizes to a peak of one so the 0.2 isolevel means something. Both cannot hold at
  once for the same array.

## The electron-phonon subsystem

`demo.py` writes no electron-phonon data at all, although producers exist under
`_demo/electron_phonon/`. The ten docstrings of `electron_phonon_self_energy` and
`electron_phonon_transport` call methods on an undefined `calculation` variable, so
they cannot run even if they were collected. This is the largest block of dead
examples left.

## The spin_texture selection

Still `_demo.band.spin_texture`: unseeded random occupations and projections on a
dispersion that is a reshaped `np.arange`.

## Quantities with no examples at all

`demo.py` writes none of `dielectric_tensor`, `elastic_modulus`,
`born_effective_charge`, `force_constant`, `internal_strain`, `piezoelectric_tensor`,
`polarization`, `electronic_minimization`, `bader` or `exciton_eigenvector`, and none
of them carries a docstring example.

These fail differently from the grid family, and more quietly. They print a fixed-width
table, so random numbers still look like output — but a random dielectric tensor need
not be symmetric, a random Born effective charge need not satisfy the acoustic sum
rule, and a random piezoelectric tensor need not respect the crystal symmetry. Nothing
in the handlers validates any of that, so there is no visible symptom at all.

## Known weaknesses in the data that already exists

- The optical phonon branches share a single modulation, a cosine of the sum of the
  reduced coordinates, and differ only in amplitude. A cosine of one integer vector
  takes five values on the q mesh, so each branch contributes five sharp lines and the
  optical part of the density of states is more structured than a real spectrum, with
  gaps no oxide has. Giving each branch its own translations, the way
  `electronic_structure.Model` disperses an electronic band, is what would fix it; a
  first attempt left one branch flat and needs its own tests.
- The `"perovskite"` selection exists only because the default selection used to pair
  the Sr2TiO4 structure with CoO's symmetry operations, so the symmetry-derived
  structure properties needed a consistent structure somewhere. The default carries its
  own I4/mmm symmetry now, so that reason is gone and the selection could be retired.
- `partial_density._correct_units` divides by the number of grid points *and* the cell
  volume, so a partial charge in VASP's convention comes out around 1e-7 while the
  documented default `current=1.0` for a constant-current image is order unity. The two
  scales cannot both be right. It has tests around it and changing it changes
  user-visible currents, so it was left alone.
