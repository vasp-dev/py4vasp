# Loose ends of describing routines for tools

`vasp.backend.inspect` describes a routine as JSON fields built from its annotations and
`_selection_axes`, and `vasp.backend.call` checks a form's values against that
description. A few things are not described yet:

- **Options passed as keywords.** `to_view` of density, potential, nics,
  partial_density and exciton.density takes `**user_options` (`isolevel`, `color`,
  `opacity`). They are untyped `**kwargs`, so `inspect` leaves them out and a form
  cannot offer them. Describing them needs either explicit keyword parameters or a
  declaration of the documented options.
- **Value ranges.** Only the cut fractions a, b, c carry a `Range` (0 to 1). Others
  have obvious bounds that a form could use for sliders or validation: supercell >= 1
  (enforced by the code), cutoff > 0, width > 0, tip_height and current > 0,
  minimum_frequency >= 0. A `Range` only describes; py4vasp does not reject values
  outside it.
- **Routines outside the form.** Only read, to_dict, to_graph, to_frame, to_numpy,
  to_view, to_contour, to_quiver, and to_stm declare their selection tokens. The
  tokens `selections()` lists for `projector.project`, `raman.activity`
  (`depolarization`), and the `select` of the electron-phonon quantities are kept as
  documented exceptions in `tests/calculation/test_selection_axes.py`.
- **The docstring parser** (`py4vasp._util.docstring`) covers the numpydoc subset
  py4vasp uses and is tested against it, but nobody checked the docstrings with
  numpydoc itself.
- **Axes per source.** `_selection_axes` is built from the default source, so a source
  with other data still offers its tokens: the DOS on the KPOINTS_OPT mesh offers atoms
  and orbitals even if VASP wrote no projections for that mesh, and the routine then
  refuses them. Likewise `selections(selection=...)` adds the axes of the default
  source. Describing the axes per source would fix both.
- **Messages written for Python users.** `call` passes the routine's messages on, and
  some do not fit a form: a value of the wrong type is "not a int | tuple[int, int,
  int] | None", "Select a single step of Dos" and "PhononBand has no steps" name
  classes, the steps count "from 0" while printouts label them from 1, and `NoData`
  advises passing the source "as the selection, e.g. selection="default"", which
  `call` forbids. The descriptions of the selection fields, written for the Python
  interface, suggest the same.
- **Pointers to text a tool does not show.** A tool shows the summary, the parameters,
  the returns and the notes of a routine, but not the docstring of its class.
  `raman.intensity` ends the description of `minimum_frequency` with "see the Notes of
  this class", which a form cannot follow. The point of those notes fits in one
  sentence: VASP stores only the magnitude of each frequency, so an unstable mode looks
  like a stable one, and `phonon.mode` reports it as imaginary. Likewise,
  `force_constant.read` says "see the class documentation for the sign convention". The
  test that keeps the prose from pointing to the Examples section
  (`test_prose_does_not_point_to_the_examples`) could also catch pointers to the class.
- **A reversed range of steps.** `call(calculation, "structure", "to_view",
  steps="::-1")` passes the check of `call`, since the range selects steps, but the
  view then raises a bare `ValueError` ("Step must be >= 1") instead of a py4vasp
  error; energy accepts the same range.
- **What a form still has to guess.** There is no call that lists the routines of a
  quantity (a tool keeps its own list); the atom axis gives the elements and the
  index range but not which indices belong to which element; `operators` is the same
  for every routine although not every routine uses arithmetic, pairs, or
  assignments; there is no way to check a selection without running the routine; and
  nothing says what a step is for a quantity (an ionic step, a QP iteration), so a
  steps selector can only be labelled "step".
