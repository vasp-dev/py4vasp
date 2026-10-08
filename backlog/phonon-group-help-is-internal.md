# `help(calculation.phonon)` shows an internal docstring

`calculation.phonon` is a `dispatch.Group` (`_calculation/dispatch.py:1062`), and
`help()` or `?` on it prints "Thin namespace for nested quantities (e.g. phonon.dos,
phonon.band). On attribute access, instantiates the dispatcher class with the source."
That is written for the developer. A user who asks what `phonon` offers should learn that
there are `band`, `dos` and `mode`, what each needs from the INCAR, and that
`force_constant` lives outside the group. The same holds for every other group.

Found alongside, in `phonon_mode.py`:

- the "See Also" section names the private path
  `py4vasp._calculation.phonon_band.PhononBand` (`phonon_mode.py:429-431`), which a user
  cannot type and which breaks if the module moves; refer to
  `py4vasp.calculation.phonon.band` instead;
- the `dispersion` source of `phonon.mode` is not documented anywhere a user reads, yet
  the missing-data message now recommends `selection="dispersion"`.

Collected from the reviewer notes of #349.
