# Loose ends from giving every quantity its selection back

`tests/calculation/test_selection_convention.py` now decides which methods take a
`selection`: those that accepted one in 0.11.3, those of a quantity with a non-default
source, and those reaching a handler method that takes one. Doing so left three points
open.

- **`structure.to_POSCAR(selection="phonon")` crashes** with `TypeError: unsupported
  format string passed to numpy.ndarray.__format__` in `_vector_to_row` on the demo
  calculation, while `lattice_vectors("phonon")` and `positions("phonon")` work and the
  `exciton` source exports fine. The phonon source was unreachable before, so nobody
  saw it; check whether the demo or the handler has the wrong shape.
- **`current_density` has no default source.** `is_available()` resolves the sole
  `"nmr"` source (`dispatch._effective_source`), but `read()` and the other methods do
  not and fail with a `FileAccessError` unless given `"nmr"`. Either let dispatch fall
  back to a sole source the same way, or let `is_available()` report `False`.
- **No page explains the sources.** A simulated user had to guess what the structure
  sources `phonon`, `exciton`, `final` and `poscar` mean and when VASP writes them; no
  docstring example passes a source. `selections()` lists every possible source, not
  the ones present (only `is_available([...])` tells), and `density.selections()`
  prints `np.str_(...)` reprs. The word "selection" also means steps in the
  structure docstrings, and `structure[...]` slices steps while `density[...]` picks
  a source.
- **Missing data gives poor errors.** `density.read("all_electron")` blames
  vaspwave.h5 and LCHARGH5 even when vaspwave.h5 exists.
- **Docstrings list `selection` first where it is the last positional argument**, e.g.
  `Structure.to_view(supercell, ion_types, selection)`, `to_molden(masses, selection)`
  and the other methods that kept their arguments for compatibility; move the entry to
  the end of Parameters. `PartialDensity.grid` has no docstring at all.
- **Steps of a single-step source are ignored.** `structure[1:3].positions("final")`
  returns one structure without complaint, because the handler drops the steps when
  the source is not a trajectory.
- **`density.to_quiver(a=0.3)` on a nonpolarized density** raises a bare
  `ValueError: shape-mismatch for sum`; probably older than this branch.

Open question found earlier: `born_effective_charge.print()` says "cumulative output"
where OUTCAR says "cummulative output" (`linear_response.F:967`, `pead.F:2356,2368`), so
grepping one for the other fails. Copy VASP's spelling, or keep the correct one?
