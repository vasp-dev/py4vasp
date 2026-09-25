# Two phonon failures point the user at the wrong thing

Found by a simulated user who had only the documentation, while validating the eV
change. Neither is a unit problem and neither was touched by that branch.

**Asking for a band structure that was never computed reads like an internal assertion.**
On a linear response run that has modes but no dispersion:

```
>>> calc.phonon.band.plot()
RefinementError: Could not understand the mode 'VaspData(None)' when refining the raw
kpoints data.
```

The word "mode" means something else entirely in a phonon context, so the reader's first
conclusion is that their phonon *modes* are broken — they are not; only the band
structure is absent. `calc.phonon.dos.read()` on the same calculation gets it right and
raises the clean `NoData` ("make sure that the provided input should produce this data").
The band path should reach the same error. The model to copy is
`phonon_mode.displace`, whose message says what happened, why, and what to change.

**A missing source blames the calculation rather than the selection.** On a dispersion
run, `calc.phonon.mode.read()` raises the generic `NoData` telling the user to check that
VASP finished, when the actual fix is `read("dispersion")` — the run is fine, the default
source simply is not the one present. `selections(only_available=True)` reports
`'phonon.mode': ['dispersion']` and would have answered it. This is the concrete phonon
instance of [missing-source-error-does-not-name-the-alternatives]; recorded here because
that note describes the class of problem and this is the case a user actually hit.

**Smaller, same area.** `python -m py4vasp convert --help` says "Specify which QUANTITY
you want to convert into which FORMAT" and lists no choices, but the only accepted value
is `structure`; every other quantity fails with `Invalid value for 'QUANTITY'`. The error
is self-correcting, the help text is not — it advertises a generality the command does
not have. There is no CLI route to phonon data at all.
