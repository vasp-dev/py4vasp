# The unit conversion factors are shared but still private, and still not CODATA

`_util/convert.py` now holds `EV_TO_THZ`, `EV_TO_CM1` and `EV_TO_MEV`, so the three
copies that used to live in `phonon_mode.py`, in a `__str__` helper, and in the demo
data generator are gone. Two things it did not settle.

**They are private.** py4vasp deliberately has no `unit=` keyword — every value it
returns is eV and a plot converts where it labels the axis. That makes the factors *more*
important to a user, not less: anyone who wants a frequency in cm⁻¹ has to convert it
themselves, and today they type a constant of their own. Measured in one workshop: a
tutorial used `8065.54` in four cells, a script shipped with it used `8065.54429`, and
py4vasp uses `8065.610420`. The spread is physically irrelevant but it shows up in the
sixth digit, which is precisely the digit that tutorial asked the reader to compare.
`optics.py:32` still keeps `HBAR_C = 1239.84` to itself for the same reason.

**They are VASP's values, not CODATA.** `EV_TO_CM1 / EV_TO_THZ` is 33.356683 where
CODATA gives 33.356410 — a relative deviation of 8.2e-6, pinned by
`tests/util/test_convert.py::test_energy_conversion_factors_are_consistent`. That was
the right call when centralising them, because a table py4vasp prints has to match the
OUTCAR it came from. It is the wrong call for a number a user does physics with, and the
two purposes want different constants.

The preferred direction is not a `py4vasp.units` module of bare floats but a real unit
library such as [Pint], so a quantity carries its unit and a conversion cannot be applied
twice or in the wrong direction. That is a larger decision than this note: it touches
every quantity, the `Graph` axis labels and the database models, and it would change what
`read()` returns. Worth scoping deliberately rather than drifting into.

Related to [energy-units-across-quantities], which lists what is not eV yet.

[Pint]: https://pint.readthedocs.io/
