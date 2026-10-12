# Metadynamics results are only in REPORT and HILLSPOT

A metadynamics run (`HILLS_BIN`, `HILLS_H`, `HILLS_W`, coordinates with status 5 in
ICONST) writes the collective variables of every step (`fic` lines) to REPORT and the
deposited Gaussians to HILLSPOT; the starting bias comes from PENALTYPOT. None of this
is in `vaspout.h5`. The transition-state tutorial (part 4, exercises 15 and 16) needs a
shell script to split the `fic` lines into one file per collective variable and a
Python script that sums the Gaussians of HILLSPOT on a grid to draw the bias potential.

This is a request to VASP first: write the collective variables of every step and the
deposited hills (position, height, width) to `vaspout.h5`. Then py4vasp could offer
the time evolution of the collective variables and the accumulated bias potential,
including a contour plot over two collective variables, as a quantity. Mind that the
bias then contains the hills of all previous runs whose HILLSPOT was copied to
PENALTYPOT, so the hills of the current run should be distinguishable.
