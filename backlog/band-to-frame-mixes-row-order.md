# `band.to_frame` pairs the energies with the wrong k-points

`Band.to_frame` returns one row per pair of k-point and band, but it lays out its
columns in two different orders:

- `kpoint_distances` and `kpoint_labels` go through `np.repeat`, so they run k-point by
  k-point, each k-point once per band;
- `bands`, `occupations`, and the projections go through `_to_series`
  (`array.T.flatten()`), so they run band by band.

With more than one band, almost every row pairs a k-point with the energy at another
k-point. Take the demo data, with 164 k-points and 8 bands. The first eight rows are
all labelled Γ at distance 0, but they hold band 1 at the first eight k-points (−5.300,
−5.302, … −5.403). Band 2 at Γ is in row 164. `to_dict` and `to_graph` are correct,
and nothing warns. Anyone who filters the frame by label or plots distance against
energy gets wrong numbers.

`tests/calculation/test_band.py` asserts the mismatch: `test_multiple_bands_to_frame`
expects `bands.T.flatten()` next to `np.repeat(distances, repeats=3)`. The tests
therefore change together with the code. Pick one order and use it for every column,
including `bands_up`, `bands_down`, and the projections. Then test that the rows of one
k-point carry that k-point's eigenvalues. The k-point-major order matches `np.repeat`
and reads as "all bands at this k-point".

The doctest of `to_frame` prints only `0  ...`, which hides the layout; printing the
first rows would show it.

Found by a simulated user who compared the frame with the raw eigenvalues.
