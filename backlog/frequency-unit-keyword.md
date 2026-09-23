# `phonon.mode` is in eV and `phonon.band` is in THz, with nothing at the call site to say so

Both are individually correct and neither is documented where a user meets it. Measured on
one BaTiO₃ calculation: the soft mode reads 0.02526 eV from `phonon.mode`, while the band
data of the same material maxes at 48.89 THz and `phonon_band.py:63` hard-codes
`g.ylabel = "ω (THz)"` with no conversion anywhere in the path. Two phonon quantities on
one `Calculation` object, two units. A workshop tutorial duly mislabeled its axes.

Vibrational spectroscopy also has an entrenched community unit, cm⁻¹, that is neither of
the two — and VASP itself is inconsistent, storing `results/phonons/eigenvalues` in eV and
`results/linear_response/raman/frequencies` in cm⁻¹. Some normalization layer is
unavoidable; doing it once here beats doing it in every notebook.

Proposal, deliberately scoped to the vibrational family only: a keyword-only
`unit="eV" | "meV" | "THz" | "cm-1"` on `phonon.mode.frequencies()`, `phonon.band` and
`phonon.dos` graphs, defaulting to today's behaviour, echoing the choice into the returned
dict (`{"unit": "cm-1", ...}`) and into the `Graph` axis label. The knowledge already
exists in `_frequency_to_string`; it is just locked inside `__str__`.

Do **not** add a global `unit=` across all 30+ quantities with an energy axis. A
half-applied option is worse than none, and questions like "does `to_dict()` change its
values, and does every downstream consumer know?" do not have one answer across the API.

Depends on [public-unit-constants]. Whatever is decided, `phonon.mode` and `phonon.band`
must at minimum state their unit in their docstrings — that part is not optional.
