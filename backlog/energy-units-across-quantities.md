# Which quantities are not in eV yet

py4vasp's policy is that every value it returns is an energy in eV, and that a plot or a
printed table converts where it labels the number. The phonon family now follows it. This
is the survey of everything else, so the next round does not have to rediscover it. None
of the items below were touched; each needs its own decision.

Pressures are settled: `read` and `print` keep the kBar VASP writes, the database stores
GPa, and `_util/convert.py` holds `KBAR_TO_GPA`. That fixed `StressModel`, which stored
kBar under GPa docstrings, and `ElasticModulusModel`, whose tensors were kBar beside GPa
moduli; `ElasticModulus.read` now states its unit.

## Wrong unit, or a value and its documentation disagreeing

- **`Stress.__str__` prints two units and `to_dict` documents neither.** The header at
  `stress.py:43` says `units (eV)` for a row divided by `eV_to_kB`; the next row is kBar;
  `to_dict` returns kBar (`stress.py:52-70`).

## No unit stated anywhere

- **`ElectronPhononSelfEnergy`** — `fan`, `debye_waller`, `self_energy()`, `energies()`,
  `eigenvalues()` (`electron_phonon_self_energy.py:45-115`). No unit in the module and no
  plot method, so there is no axis label to cross-check. Physically eV; confirm with VASP
  before writing it down.
- **`ElectronPhononInstance.read_metadata()["selfen_delta"]`**
  (`electron_phonon_instance.py:49`) — a broadening, presumably eV.
- **`ExcitonEigenvector.to_dict()["bands"]`** (`exciton_eigenvector.py:40-49`) — eV by
  construction (shifted by the Fermi energy) but never said.
- **`transport_function` is missing from `electron_phonon_transport.py:37-43`'s `UNITS`**,
  while `read()` returns it (`:100-108`). `_SeriesBuilderBase.ylabel` indexes that dict,
  so the quantity is unplottable rather than mislabelled.
- **`Bandgap`, `Workfunction`, `DielectricFunction.to_dict`,
  `ElectronPhononChemicalPotential.to_dict`** all return eV and say so only on the axis
  label, not in the method docstring.

## A blanket "everything is eV" cannot hold here

- **`Energy.to_numpy("temperature TEIN")` returns Kelvin** from the same method whose
  other selections return eV (`energy.py:38,58,139-160`). `to_graph` puts it on a second
  axis, or on the primary one if selected alone (`energy.py:557-558`). Deciding whether
  temperature belongs in the `energy` quantity comes before deciding its unit.
- Other Kelvin quantities are consistent and documented:
  `ElectronPhononTransport.temperatures()`, `ElectronPhononBandgap`'s `temperatures` key.

## Found while working nearby, not unit related

- **`DispersionHandler.to_database` crashes on absent eigenvalues.** `_dispersion.py:48`
  guards with `check.is_none`, but `_spin_polarized` at `:96` reads `.ndim` on the raw
  data first and raises `NoData`. The `None` branch is therefore dead code and
  `DispersionModel(eigenvalue_min=None)` is unreachable through this path. Shared by the
  electronic band, so it wants its own fix.
- **`force_constant.py:19` names Å-per-Bohr `_A_TO_BOHR`.** The usage
  `positions / _A_TO_BOHR` at `:151` is numerically right, so this is a naming bug only.
  The Bohr coordinates are the `[FR-COORD]` block of `to_molden`, which the molden format
  defines in Bohr, so they are not in conflict with the eV/Å² that `__str__` prints for
  the constants. Left deliberately untouched when `to_molden` was fixed to write its
  `[FREQ]` block in cm⁻¹ from the mass-weighted dynamical matrix, which kept that change
  to the frequencies and the normal modes alone.
- **`electron_phonon_chemical_potential.py:38-48` looks like it has two table headers
  swapped**: `chemical_potentials` prints under "Number of electrons per cell" and
  `carrier_densities` under "Chemical potential".
- **`dielectric_tensor.py:279`** says `including local field effects in RPA (Hartree)`.
  The parenthesis reads like a unit in a grep, but the tensor is dimensionless
  (`dielectric_tensor.py:60`) and this is the approximation, not the unit. Leave alone.

Depends on [public-unit-constants] for whatever the conversion factors end up being.
