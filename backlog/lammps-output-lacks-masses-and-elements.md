# The LAMMPS output does not say which element each atom type is

Found by a simulated user while validating the removal of `convert --selection`, whose
goal was to start a classical MD run from the structure VASP relaxed.

`Structure.to_lammps()` (and so `py4vasp convert structure lammps`) writes atom types
1, 2, 3 but neither a `Masses` section nor a comment mapping the types to elements. For
Sr2TiO4 the user had to recover the order Sr, Ti, O from `print(calculation.structure)`
and type the masses into the LAMMPS input by hand.

Writing a `Masses` section (standard atomic weights, the same defaults the phonon
methods use) and an element comment per type would make the file usable as it is. See
also `mass-overrides-per-element.md` for how masses are exposed elsewhere.

Unverified side note from the same trial: the demo box has a tilt xy = 4.69 against
lx = 6.92, i.e. |xy| > lx/2, which older LAMMPS versions reject unless
`box tilt large` is set. Check whether the standard form should reduce the tilt.
