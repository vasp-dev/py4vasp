# `convert --from vaspwave.h5` silently converts a different structure

Found by a simulated user while validating the removal of `convert --selection`.

A demo calculation contains `vaspout.h5` and `vaspwave.h5`. Pointing the command at the
wrong one is an easy slip:

    $ py4vasp convert structure lammps --from example/vaspwave.h5

It succeeds without a warning and prints a plausible geometry that differs from the one
of `--from example` by about 1e-5 Å in the cell (xhi 6.9229000000000003 against
6.9229006367598549). Nobody would notice by eye.

Decide what structure `vaspwave.h5` holds and whether reading it as a calculation is
intended. If it is not, the command should reject the file and name `vaspout.h5`; if it
is, the help should say what the user gets.
