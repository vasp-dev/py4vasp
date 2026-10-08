# Colors are picked ad hoc instead of from qualitative, diverging and sequential schemes

py4vasp has one palette, `_config.VASP_COLORS`: seven brand colors (`purple`, `cyan`,
`blue`, `red`, `gray`, `dark`, `green`) in one dict. That dict is used for three
different jobs without saying which one is meant:

- **as a qualitative cycle.** `graph.py:50` puts all seven into the plotly template's
  `colorway`, while `Graph._generate_plotly_traces` (`graph.py:477-482`) cycles through
  them *without* `dark`. So a figure built by plotly and the same figure built by `Graph`
  color their series differently. Neither cycle was chosen for being told apart, by a
  color-blind reader or in greyscale print: it is simply the dict in insertion order.
- **as a diverging pair.** `density.py:272-273` and `nics.py:196-197` draw the positive
  isosurface blue and the negative one red. The plotly template's contour scale
  (`graph.py:45-48`) runs *red → white → blue*, while `Contour`'s fallback
  (`contour.py:600-605`) runs *blue → white → red* and its `"diverging"` scheme uses
  plotly's `RdBu_r`. That is three diverging scales, two of them pointing opposite ways.
- **as a single highlight.** `cyan` is the default isosurface color in `density`,
  `potential`, `partial_density` and `exciton_density`. Meanwhile `view.py:30` defaults
  to `#2FB5AB`, a cyan not in `VASP_COLORS`, and `electronic_minimization.py:52` has
  its own `_UNUSUAL_COLOR = "#4d4d4d"`.

`Contour.color_scheme` (`contour.py:142-158`) comes closest to a structure: `"auto"`
chooses `diverging`, `positive` or `negative` from the sign of the data, and the user
can pick `sequential` or `monochrome`. But it mixes brand colors with plotly built-ins
(`Reds`, `Blues_r`, `Viridis`, `RdBu_r`, `turbid_r`), it exists only on `Contour`, and
`positive`/`negative` are really sequential schemes under another name.

## Proposal

Follow Paul Tol's approach ([Tol]): give each *kind* of data its own scheme, chosen to
stay distinct for color-blind readers and in greyscale, and use it everywhere that kind
of data is drawn.

- **Qualitative**: for series that have no order, e.g. the default cycle of `Graph`,
  projections in `dos`/`band`, calculations overlaid in a convergence study. A fixed,
  ordered list, with the brand colors adjusted or reordered until neighbours can be
  told apart.
- **Diverging**: for signed data with a meaningful zero, e.g. density differences,
  NICS, ± isosurfaces, `Contour` with `"diverging"`. A single orientation (decide once
  whether positive is red or blue), and the same pair of end colors for the 2D colormap
  and the 3D isosurfaces.
- **Sequential**: for magnitudes, e.g. STM images, `positive`/`negative` contours,
  densities. A perceptually uniform scale, with the `negative` variant reversed rather
  than a separate palette.
- Optionally a **bad-data color** for NaN/masked values, which Tol keeps separate from
  every scheme.

Where it could live: a small `_config` (or `_util/colors.py`) module exposing the three
schemes by name, `VASP_COLORS` kept as the brand source the schemes are built from, and
every hard-coded literal above replaced by a lookup. `Contour._get_colormap_themes`
then maps its options onto those schemes instead of plotly names, and the plotly
template takes its `colorway` and contour scale from the same place, so the two paths
agree.

Things to decide before starting:

- whether the brand colors are kept as they are, adjusted towards a tested palette, or
  used only as accents next to one;
- the names a user sees. `Contour.color_scheme` already accepts `"positive"`,
  `"negative"`, `"monochrome"` and `"stm"`, so the new names need aliases or a
  deprecation path;
- how far it reaches: plotly graphs only, or also the defaults of `view`
  (isosurfaces, arrows) and the VASP Viewer config.

Changing default colors changes every figure in the documentation and every image a
user regenerates, so it should land as one deliberate change rather than piecemeal.

Related: [batch-combine-spectral-quantities] wants one color per calculation in an
overlay, which is the qualitative scheme applied to whole graphs.

[Tol]: https://personal.sron.nl/~pault/
