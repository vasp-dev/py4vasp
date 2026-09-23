# Several features exist, are exactly what users want, and are found by nobody

A review of a four-notebook workshop tutorial found four places where the tutorial
hand-rolled code that py4vasp already does — in two cases getting the physics wrong in the
process. The features are not missing. They are undiscoverable.

**`neighbor_list`.** Two cells computed neighbour distances with
`d - np.rint(d)`. That minimum-image convention is only valid for near-orthogonal cells
and the cells here were 120° hexagonal, so 8 of 72 atoms in the 6×6×1 supercell got the
wrong distance — 10.682 Å reported where the truth is 6.484 Å. Every
force-constant-versus-distance plot in that section has misplaced points at large
distance, in a section whose entire message is "force constants decay with distance".
`calculation.neighbor_list.read(cutoff=...)` already returns indices, distances, distance
vectors and cell offsets, and `_replica_counts` (`neighbor_list.py:47`) uses perpendicular
cell widths, so it is correct for tilted cells. It removed ~12 lines per cell and fixed
the bug.

**`phonon.band.to_view()`.** See [phonon-mode-visualization] — a tutorial installed a
third-party package and round-tripped through an external website rather than use it.

**`dielectric_function`'s selectors.** Three cells reduced a 3×3 tensor to an isotropic
average by hand, `(eps[0,0] + eps[1,1] + eps[2,2]) / 3`, although `to_dict()` and
`to_graph()` already offer `isotropic`, `xx`, `yy`, `zz`, `xy`, `xz`, `yz`
(`dielectric_function.py:181`).

**`Graph.__add__` and `Graph.label`.** See [batch-combine-spectral-quantities].

Two of the tutorial's four quantitative physics errors would have been impossible had the
author found the call that was already there. That makes this a documentation problem with
a correctness cost, not a cosmetic one: the how-to guides and the quantity docstrings need
to answer the question the user actually arrives with ("how far apart are these atoms?",
"how do I see this mode move?", "how do I overlay these four runs?") rather than only
describe the method they would need to already know the name of.

Related: [user-facing-documentation].
