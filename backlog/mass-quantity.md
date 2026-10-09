# `calculation.mass` only knows the standard atomic weights, not the masses VASP used

`calculation.mass` is a plain class that exposes `STANDARD_ATOMIC_WEIGHTS`, the
defaults `force_constant` and `phonon.mode` fall back to when `masses=` is not given.
It reads nothing from the calculation. VASP uses the POMASS of the POTCAR instead, so
a user who overwrote it, e.g. to study an isotope, has to remember to pass the same
masses again, and `phonon.mode` silently undoes the mass weighting with the wrong
masses if they forget.

Turn `Mass` into a quantity:

- `raw.Mass` linking `stoichiometry` (sources `default` and `phonon`, the primitive
  cell of the dispersion), plus the POMASS once VASP writes it to vaspout.h5;
- `calculation.mass.read()` returning the element and mass of every atom; `print`
  with one line per ion type;
- once POMASS is available, make it the default of `masses=` so py4vasp matches the
  OUTCAR without the user typing anything.

A registered quantity (`@quantity`) has to implement `read`, `print`, `__str__`,
`_repr_pretty_` and `selections` (enforced by `tests/calculation/test_print.py` and
`test_all_quantities_implement_read`), and needs a `_demo` producer, a
`RawDataFactory` method in `tests/conftest.py` and an entry in `tests/test_doctest.py`.
Keep `STANDARD_ATOMIC_WEIGHTS` as a class attribute, and drop the `mass` class
attribute on `Calculation` together with the static entry in
`docs/calculation/_index.rst` once the registry provides it.
