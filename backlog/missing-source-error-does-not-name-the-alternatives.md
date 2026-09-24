# A missing default source reports "did VASP finish?" instead of naming the sources

`calc.phonon.mode.plot()` on a calculation that only postprocessed force constants
raises

    NoData: Could not find data in output, please make sure that the provided input
    should produce this data and that the VASP calculation already finished. Also check
    that VASP did not exit with an error.

All three suggestions are wrong: the data is there, VASP finished, and nothing errored.
The calculation simply stores its modes under a different source. In the very same
session `calc.phonon.mode.is_available(["default", "dispersion"])` returns
`{'default': False, 'dispersion': True}` — py4vasp knows the answer and does not say it.

A simulated user hit this on their first attempt at the more common calculation (a
dispersion) and went off to debug their INCAR. They ranked it "would have stopped me".

This is not specific to phonons: every quantity with more than one source behaves this
way — `density` with `kinetic_energy`, `band`/`dos` with `kpoints_opt`, `structure` with
`final`. The fix belongs in the dispatch layer, which knows both the quantity and its
schema sources: when access to a source fails with `NoData`, re-raise naming the sources
that *are* available, e.g.

    NoData: 'phonon_mode' has no data for the source 'default' in this calculation.
    Available sources: 'dispersion'. Select one, e.g. calc.phonon.mode.plot("dispersion").

Touching `_dispatch` affects all ~40 quantities, so it wants its own change and its own
tests rather than riding along with a feature.

Related: in a plain Python session the same condition prints five to eight frames of
py4vasp internals before the message. The `interactive` extra is documented to condense
errors; a script user gets the full traceback.
