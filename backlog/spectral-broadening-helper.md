# py4vasp has no broadening helper, so every spectrum hand-rolls a Lorentzian

`grep -rl "fwhm\|broaden" src/py4vasp/` hits `_raw/data.py`, `electron_phonon_transport.py`
and the demo data generators — there is no shared function that turns a set of discrete
(position, weight) pairs plus a width into a spectrum.

Every consumer therefore writes its own. In one workshop tutorial two cells produced
visibly different spectra of the *same* benzene data purely because they chose different
FWHM, normalization and cutoffs, and a third implementation in the shipped
`raman_plot.py` normalized its Lorentzian to unit area where the notebook cells
normalized to unit peak height.

A small `_util` helper taking positions, weights, a grid, a width and a shape
(Lorentzian/Gaussian), with the normalization convention stated in the docstring, covers
the phonon DOS, IR and Raman spectra, and the broadening already open-coded in
`electron_phonon_transport.py`. Worth landing on its own, before
[raman-quantity] needs it.
