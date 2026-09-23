# `phonon.band` can animate modes but `phonon.mode` cannot be plotted at all

`PhononBandHandler.to_view` (`phonon_band.py:73`) builds a `view.PhononDispersion` with
eigenvectors, frequencies, q points and path labels, so `calc.phonon.band.to_view()`
animates modes in the notebook. `phonon.mode` has `read`, `to_dict`, `frequencies`,
`print` and `__str__` — and no `plot` or `to_view`.

That asymmetry costs users the obvious thing. A workshop tutorial that wanted to *look* at
the Γ-point modes of graphene installed the third-party `phononweb` package, had it reopen
`vaspout.h5` itself, wrote a JSON file, then had the reader download it and upload it to
an external website — four cells and a round trip out of the notebook, for something
`phonon.band.to_view()` already does one section earlier in the same tutorial. A different
tutorial answered "which modes are unstable at Γ, X and M" from a numeric table because
there was no way to animate the mode it had in hand.

`phonon.mode.plot()` / `to_view()` animating the eigenvector of a selected mode is the
missing half of the pair. `phonon_band.py`'s `view.PhononDispersion` construction is the
model to follow; the data needed is already in `raw.PhononMode`.

Caveat worth checking first: `phonon.band.to_view()` needs
`results/phonons/primitive/position_ions` in the output and raises `NoData` without it, so
confirm which VASP versions can supply the equivalent for `phonon.mode`.

A second small addition fits naturally in the same class:
`phonon.mode.degenerate_groups(tol)`, grouping modes by frequency within a tolerance. It
is generic, and [raman-quantity] needs it.
