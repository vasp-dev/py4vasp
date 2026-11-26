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
- **What a form still has to guess.** There is no call that lists the routines of a
  quantity (a tool keeps its own list); the atom axis gives the elements and the
  index range but not which indices belong to which element; `operators` is the same
  for every routine although not every routine uses arithmetic, pairs, or
  assignments; and there is no way to check a selection without running the routine.
