# A combined `Graph` has no `to_image`, so the comparison figure cannot be saved

`docs/_index.rst` tells users the `plot` extra enables "``plot``, ``to_plotly``,
``to_image``, ``to_frame``, ``to_csv``". Those live on the *quantities*
(`graph.Mixin`), not on `Graph`. So the moment a user does the thing the
documentation now recommends -- combine two calculations into one figure with
`+` -- they lose the documented way to save it:

```python
graph = a.dos.plot().label("coarse") + b.dos.plot().label("dense")
graph.to_image("comparison.png")   # AttributeError
```

`Graph` has `to_plotly`, `to_frame`, `to_csv` and `show`, so `to_image` is the one
gap, and `graph.Mixin.to_image` is a three-line wrapper around
`to_plotly().write_image(...)` that could move onto `Graph` with the Mixin
delegating to it. The `AttributeError` currently suggests `to_frame`, which sends the
user somewhere else entirely.

A user simulation found this immediately: it is the single most likely next step after
overlaying two calculations, and the only one of the three tasks in that trial that
came back "partly" rather than "yes".

Related: [batch-combine-spectral-quantities] wants per-calculation colours in the same
overlay workflow.
