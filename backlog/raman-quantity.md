# There is no Raman support at all

`grep -ril raman src/` returns nothing. VASP writes `results/linear_response/raman`, so
anyone doing Raman spectroscopy drops out of py4vasp and opens the file with `h5py`.

Measured cost in one workshop tutorial: three notebook cells import `h5py` and walk raw
HDF5 paths, and the exercise ships a 458-line `raman_plot.py`. The powder-averaging
routine is written out four times in that one tutorial, and two of the copies already
disagree on the Lorentzian normalization — one normalizes to unit area, the other to unit
peak height — which is exactly the silent divergence a library prevents. The idiom
`rt[..., 0] + 1j * rt[..., 1]` appears four times although `_util/convert.py:30` already
has `to_complex`.

A `calculation.raman` quantity should expose `raman_tensor` (nmodes, 3, 3, nedos, complex),
`frequencies` and `energies_dielectric_function`, following `dielectric_function.py` and
`optics.py` for house style. On top of the reader:

- powder/invariant averaging through the existing selection grammar, so `component=None`
  → powder and `"xy"` → an oriented element map onto py4vasp selection strings;
- the Bose factor as `temperature=`, and the (ω_L − ω)⁴ Stokes prefactor;
- `to_graph(laser=..., broadening=..., temperature=...)`, plus an
  `excitation_profile()` parallel to `optics.absorption_graph()`;
- `__str__` printing a mode table, and `to_database()` returning a `RamanModel` in
  `_raw/models.py` alongside `DielectricFunctionModel`.

Two decisions to settle here rather than in each notebook. First, `activity` needs the
symmetric part of a *complex* tensor: `0.5 * (R + R.T)` (what every current implementation
does) or `0.5 * (R + R.conj().T)`. Second, the unit of `frequencies` — eV is consistent
with `phonon.mode`, cm⁻¹ is what the file stores and what every Raman consumer wants.
Whichever is chosen, the docstring has to say it; see [frequency-unit-keyword].

This is large enough to need chunking. The broadening helper it depends on is
[spectral-broadening-helper]. The degeneracy grouping it needs
(`phonon.mode.degenerate_groups(tol)`) is generic rather than Raman-specific and can land
here or separately.
