# Selection errors quote the parsed text, not what the user typed

`dispatch._parse_selections` splits the source off a selection and hands the handler the
rest, rebuilt with `select.selections_to_string`. A handler that rejects the rest can
only quote that rebuilt text. So `elastic_modulus.voigt("relaxed-ion")` answers
"The selection 'relaxed - ion' is not one of ...", which reads as if py4vasp computed a
difference, and `voigt("relaxed ion")` complains about `'relaxed'` alone. The message
still lists the valid choices, so the user recovers, but every quantity whose handler
validates its selection has the same echo.

Pass the user's original text along with the parsed remainder (e.g. a third field on
`SelectionContext`) so that error messages can quote it verbatim.

Related: `selections()` now lists the tokens of the routines a form offers (read,
to_graph, to_view, to_contour, ...), as each quantity declares them in
`_selection_axes`. Choices of other routines, such as `clamped_ion` and `relaxed_ion` of
`elastic_modulus.voigt` and the moduli, are still discoverable only from the docstrings
or from an error message.

Found by the simulated user that validated `ElasticModulus.voigt` and the moduli.

More uneven messages, found by a simulated user of the selections:
- A misspelled source (`kpoint_opt(...)`) or a lowercase element (`au(d)`) dumps every
  atom index and orbital, close to 900 characters, never lists the sources, and does not
  suggest the close match.
- `up` on a calculation without spin polarization suggests "Did you mean 'p'?" instead
  of saying that only `total` exists.
- `structure.read("fianl")` passes the typo to the ion types and then advises
  `selection="fianl"`.
- Out-of-range steps are quoted as Python objects, e.g. `slice(120, 130, None)`.

Found by a simulated user of the projections of dos and band:
- An atom index outside the structure, `8(d)` in a cell of 7 atoms or `0(d)` (indices
  start at 1), answers "Could not parse the selection "8", please check the spelling
  and capitalization" instead of saying how many atoms there are.
- VASP's own name for an orbital, `x2-y2` as in the PROCAR, is split at the minus sign:
  `Ti(x2-y2)` complains about "x2" and suggests "fzx2". `Ti(dx2-y2)` does get the
  right suggestion, `dx2y2`.
- `3(d)` is labelled `Ti_1_d` in the result. The label counts within the element, the
  selection counts in the order of the POSCAR, so `Ti_1` reads like atom 1.
