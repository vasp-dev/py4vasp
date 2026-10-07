# Changing the mass of one element means typing every mass by hand

`force_constant.frequencies`, `displacements` and `to_molden`, like `phonon.mode`'s
`displace`, `displacements` and `to_view`, take `masses=` as one number per atom. Their
docstrings say the default is "the standard atomic weight", but nothing public returns
those numbers. A simulated user studying ¹⁸O in Sr2TiO4 had to copy 87.62 / 47.867 /
15.999 out of a docstring example to build a seven-entry list in which only the oxygen
changed.

The first thing that user tried was `masses={"O": 17.999}`, which py4vasp rejects with
a clear message. It is the natural idiom for an isotope substitution and covers the
common case with one entry.

Two options, not exclusive:

- accept a mapping from element to mass in `_util.masses.resolve` and override only
  those elements, keeping the per-atom sequence for site-specific substitution;
- expose the default masses, e.g. through the structure, so a user can start from them.

Either touches both quantities at once, because they share `resolve`. Found by the
user simulation for the force-constant frequencies branch.
