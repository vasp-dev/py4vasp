# Passing ε∞ and Z* to a polar phonon run still takes two calls

`calculation.dielectric_tensor.to_INCAR()` and `calculation.born_effective_charge.to_INCAR()`
each produce one INCAR tag in the orientation VASP reads (`PHON_DIELECTRIC` and
`PHON_BORN_CHARGES`). The FAQ in `DefaultCalculationFactory` shows the user joining
the two strings and appending them to the INCAR. That works, but a polar phonon run
always needs both tags, plus `LPHON_POLAR = .TRUE.`, so every user writes the same
three lines and can forget one of them.

There is no `calculation.phonon` quantity to hang a helper on: `_calculation/phonon.py`
only holds shared docstring text, and `phonon` is a group (`phonon.band`, `phonon.dos`,
`phonon.mode`). Options worth weighing:

- a method on the `Calculation` itself, e.g. `calculation.generate_polar_phonon_tags()`,
  which is a verb per the naming convention for input-file content;
- a group-level method once `phonon` groups can carry methods of their own;
- extending `py4vasp.control.INCAR` (`_control/incar.py`) so it can merge tags into an
  existing file, which would also help the KPOINTS and POSCAR writers.

Things a helper should settle that the two methods leave to the user:

- **Primitive cell.** VASP expects Z* for the atoms of the primitive cell of the phonon
  run and stops if the count differs (`polar_ewald.F:93-100`). A helper that also sees
  the phonon structure could check the count and the ion order before VASP does.
- **Which ε.** `DielectricTensor.to_INCAR` defaults to the clamped-ion ε∞; a helper
  should not expose the other choices.
- **Source run.** The tensors usually come from a different calculation than the one
  the INCAR is written for, so the helper probably takes the linear-response
  calculation as an argument.
- **Length of the tag.** VASP counts the elements of a tag in a
  `character(len=32767)` work buffer (`incar_reader.F:313`, used by
  `count_elements(INCAR, "PHON_BORN_CHARGES")` in `phonon.F:220`). `to_INCAR` writes
  about 126 characters per ion (with the wider gap between the rows of each tensor), so
  somewhere around 260 ions in the primitive cell the
  count may be truncated and VASP stops with "not divisible by 9". Not tested against
  VASP; a helper could warn, or `to_INCAR` could write fewer digits.
