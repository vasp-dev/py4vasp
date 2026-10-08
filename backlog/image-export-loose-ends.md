# Image export: loose ends after `Graph.to_image`

#350 made `Graph.to_image` work and gave both routes the same extension check. What is
left differs between the two routes or is missing from both:

- **Positional vs keyword.** `Graph.to_image(filename)` takes the filename positionally;
  a quantity's `to_image(*args, filename=None)` passes positional arguments on to
  `to_plotly`. So `calculation.dos.to_image("dos.png")`, the natural call after using
  the graph version, fails with "Could not parse the selection "dos.png" … py4vasp
  supports any of the following selections: "Sr", "1", …" and never mentions
  `filename=`. The parser could recognize a supported image extension and point to the
  keyword. Same for `to_csv`.
- **Relative paths mean different things.** For a `Graph` they are relative to the
  current directory, for a quantity relative to the calculation directory (see also
  [files-saved-next-to-a-temporary-demo-vanish]).
- **Paths are not normalized.** `"~/fig.png"` is not expanded, and a missing parent
  directory surfaces as plotly's `FileNotFoundError`, not a py4vasp message.
- **No size control.** `write_image` takes `width`, `height` and `scale`, but neither
  `to_image` forwards them, so a print-quality figure needs `to_plotly()`.
- **Empty top margin.** A saved image keeps the space plotly reserves for a title the
  figure does not have.
- `Mixin.to_image` still calls `to_plotly().write_image` instead of delegating to
  `Graph.to_image`, so fixes to one route have to be made twice
  (`mixin.py:105-110`, `graph.py:758`).

Collected from the reviewer notes of #350 and from probing it.
