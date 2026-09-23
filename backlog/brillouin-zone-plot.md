# There is no way to draw a Brillouin zone, a q/k mesh or a high-symmetry path

Every tutorial that explains sampling needs the same picture: the first Brillouin zone,
the Γ-centred mesh inside it, and the labelled path through the high-symmetry points.
py4vasp cannot draw any of it, so a workshop shipped a 124-line matplotlib script that
rebuilds the reciprocal lattice by hand — and a second one for the direct lattice.

`py4vasp.calculation.kpoint` already holds the pieces: `mode`, `line_length`, `labels`,
`distances`, and the cell. A `kpoint.plot()` / `to_view()` that draws the zone, the
sampled points and the path would sit naturally next to them, and py4vasp already
generates k-paths and k-meshes from the CLI, so the same picture serves as a check on
what was generated.

The existing script is 2D-hexagonal-specific (it uses `r = |b1 + b2| / 3`), so the real
work is the general case: a Wigner–Seitz construction of the zone from the reciprocal
lattice, in 3D, in a form the `view`/`graph` layer can render. That is the part to scope
before starting.
