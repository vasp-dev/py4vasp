# Moving ε∞ and Z* into the next run's INCAR is hand-formatted by the user

Any polar-phonon workflow — LO-TO splitting, IR intensities, polar interpolation — reads
the dielectric tensor and the Born effective charges out of one run's `vaspout.h5` and
writes them into the next run's INCAR as `PHON_DIELECTRIC` and `PHON_BORN_CHARGES`. py4vasp
supplies the numbers and then stops.

So a workshop tutorial hand-built the blocks with f-strings: continuation backslashes,
digit counts, ion order, and a row/column choice. That last one is the trap — the cell
printed `dielectricTensor[:, i]` and `b[:, i]`, i.e. the transpose of py4vasp's storage
order. MgO is diagonal so nothing showed. Z* is not symmetric in general, and a reader
copying the recipe to a low-symmetry polar material gets a silently transposed Z*.

Tag names, block syntax, orientation and the digits needed are VASP-format knowledge,
which is py4vasp's job. `py4vasp.control.INCAR` (`_control/incar.py`, with
`from_string`/`write`) is the natural home for the writing end, and the quantity classes
already have `to_string`/`__str__` conventions for the producing end:

- `born_effective_charge.to_INCAR()` → the `PHON_BORN_CHARGES` block, all ions, ordering
  matching the POSCAR, orientation verified against what VASP parses;
- `dielectric_tensor.to_INCAR(selection="clamped_ion")` → the `PHON_DIELECTRIC` block;
- optionally `calc.phonon.polar_incar_tags()` returning both, so the whole step becomes
  two lines that can write straight into the next calculation's directory.

Verifying the orientation VASP expects is the part that needs care, and is exactly the
part every user currently guesses at.
