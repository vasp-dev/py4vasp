# `selections()` of the projected quantities is undocumented

A simulated user who only read the documentation found:

- `dos.selections`, `band.selections`, and `projector.selections` have no docstring,
  while `structure.selections` has one; the argument that picks a source is
  discovered only by experiment.
- The return value has four shapes: sources plus atom/orbital/spin (dos, band), parts
  only (projector), sources only (structure), and a flat list of pairs
  (neighbor_list, reaction_path).
- The sources listed are those py4vasp knows, not those the calculation contains;
  `Calculation.selections(only_available=True)` gives the latter, but no docstring
  says to combine the two. `qp_linearized` and `qp_zeros` are not explained.
- f orbitals are offered on a run whose projections stop at d (LORBIT = 11 with 9 lm
  columns), and selecting them returns arrays of zeros, which reads like physics
  ("no f character") rather than "not computed".
