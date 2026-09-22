# Unit conversion factors are private, so users hardcode their own

`phonon_mode.py:78-81` defines `eV_to_THz = 241.798934781` and `eV_to_cm1 = 8065.610420`
inside `_frequency_to_string`, a `__str__` helper. `optics.py:31` defines
`HBAR_C = 1239.84`. None of it is reachable from outside, so every user who needs a
frequency in cm⁻¹ or THz types a constant of their own.

The consequence, measured in one workshop: a tutorial used `8065.54` in four cells, a
script shipped with it used `8065.54429`, and py4vasp uses `8065.610420`. CODATA gives
8065.543937. The spread is physically irrelevant but it means a frequency printed by
`print(calc.phonon.mode)` and the same frequency converted by hand differ in the sixth
digit — precisely the digit that tutorial then asked the reader to compare.

Add one authoritative CODATA value each in a public place — `py4vasp.units`, or additions
to `_util.convert` (`eV_to_THz`, `eV_to_cm1`, `eV_to_meV`, `eV_to_nm`) — and make
`_frequency_to_string` and `optics` use them instead of their private copies. Purely
additive, no API risk, and it is the prerequisite for [frequency-unit-keyword].
