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
- **A positional source on `structure.read` still lands in `ion_types`.** For
  compatibility with 0.11.3 `selection` comes after `ion_types`, so `read("final")`
  names the ions after the letters of "final", as it did then. A string `ion_types`
  matching a source name could raise `IncorrectUsage` pointing to `selection=`.

Open question found earlier: `born_effective_charge.print()` says "cumulative output"
where OUTCAR says "cummulative output" (`linear_response.F:967`, `pead.F:2356,2368`), so
grepping one for the other fails. Copy VASP's spelling, or keep the correct one?
