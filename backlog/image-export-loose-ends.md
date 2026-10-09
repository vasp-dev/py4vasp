# Image export: loose ends after `Graph.to_image`

#350 made `Graph.to_image` work and gave both routes the same extension check. What is
left differs between the two routes or is missing from both:

- **Relative paths mean different things.** For a `Graph` they are relative to the
  current directory, for a quantity relative to the calculation directory (see also
  [temporary-demo-calculation-loose-ends]).
- **Paths are not normalized.** `"~/fig.png"` is not expanded, and a missing parent
  directory surfaces as plotly's `FileNotFoundError`, not a py4vasp message. A shared
  helper for both routes is a few lines, but the mixin tests write to a `Path("folder")`
  that does not exist, so a directory check means rewriting them on `tmp_path`.
- **No size control.** `write_image` takes `width`, `height` and `scale`, but neither
  `to_image` forwards them, so a print-quality figure needs `to_plotly()`.
- **Empty top margin.** A saved image keeps the space plotly reserves for a title the
  figure does not have.
- `Mixin.to_image` still calls `to_plotly().write_image` instead of delegating to
  `Graph.to_image`, so fixes to one route have to be made twice
  (`mixin.py:105-110`, `graph.py:758`).

A filename passed positionally to a quantity's `to_image`/`to_csv` now raises
`IncorrectUsage` naming `filename=`.

Collected from the reviewer notes of #350 and from probing it.
