# `Batch` covers only energies, forces and stresses, so every convergence study hand-builds plotly

`_combine/` contains `energies.py`, `forces.py` and `stresses.py` and nothing else. The
most common thing anyone does with several calculations — overlay the same spectrum from a
k-mesh, q-mesh or supercell-size scan — has no support, so users drop to raw plotly.

What that produces in practice, from one workshop tutorial:

- a q-mesh DOS overlay where about half the cell is dead code: `make_subplots`,
  `final_fig`, `sample_colorscale` and the colour list are all computed and discarded, so
  the intended viridis colouring is silently not applied, and the axis labels
  `phonon.dos.to_graph()` already sets are retyped by hand;
- a band-structure overlay that calls `to_plotly()`, then reaches into `fig.data[i]` to
  set colours, picks the colour by string-matching a label it had just assigned itself,
  and shadows its own loop variable;
- three near-identical fetch loops in one notebook, each re-reading the same HDF5 files.

Most of this needs no new feature — `Graph.__add__` (`graph.py:270`) merges series *and*
fields via `_merge_fields`, and `Graph.label` (`graph.py:369`) relabels. What is genuinely
missing is two things: extending `Batch` (or a small `py4vasp.plot_batch(quantity,
**paths)`) to `phonon.dos`, `phonon.band` and `dos`; and a way to give each calculation in
the overlay one colour, since `Series` has a `color` field but nothing propagates a
per-graph colour when graphs are added.

With both, a convergence study is a comprehension over paths plus a `sum`, and the plotly
escape hatch closes for this and for every other benchmarking workflow.
