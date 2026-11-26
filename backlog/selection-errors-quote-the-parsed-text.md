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
