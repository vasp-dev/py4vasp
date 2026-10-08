# A frozen-phonon structure for another isotope has to be built by hand

`force_constant.frequencies` and `force_constant.displacements` take `masses=`, so a
user can see how ¹⁸O shifts the modes of Sr2TiO4 without rerunning VASP. Turning such
a mode into a displaced structure for a frozen-phonon calculation is not supported:
`phonon.mode.displace` exists only for the modes VASP computed, and its `masses` must
be the POMASS VASP used, because it only undoes the mass weighting of the stored
eigenvectors.

A simulated user studying ¹⁸O tried `phonon.mode.displace(masses={"O": 17.999})`
first. On a run with the standard oxygen mass this yields a pattern that is no normal
mode at all (the centre of mass moves), and an ordinary user would not notice. The
documentation now says so and points to `force_constant`, but the only route to a
POSCAR is to take `force_constant.displacements(masses)[mode]`, add it to the
positions via `structure.to_ase()` and convert back with `Structure.from_ase`.

Options:

- `force_constant.displace(selection, amplitude, masses)` mirroring
  `phonon.mode.displace`, with the same amplitude convention (needs the frequency of
  the recomputed mode, which `force_constant.frequencies(masses)` provides);
- or let `phonon.mode.displace` warn when `masses` differ from the ones the
  eigenvectors are weighted with, once VASP writes POMASS (see
  `mass-quantity.md`).

Found by the user simulation for the mass-overrides-per-element branch.
