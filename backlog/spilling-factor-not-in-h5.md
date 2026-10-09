# The spilling factor of an MLFF run is only in ML_LOGFILE

An MLFF production run (`ML_MODE = run`) reports the spilling factor every
`ML_OUTBLOCK` steps in ML_LOGFILE (`SFF` lines) and stops once it exceeds the limit.
It is the signal for when the force field extrapolates, and the structures with an
intermediate spilling factor are the ones worth adding to the training set. The
transition-state tutorial (part 4, exercises 15 and 16) parses the `SFF` lines by hand
to plot the spilling factor, and a separate script picks the structures with a
spilling factor between two thresholds from XDATCAR and writes them to
POSCAR.interactive for retraining.

This is a request to VASP first: write the spilling factor of every reported step to
`vaspout.h5`. Then py4vasp could plot it against the step and select the steps whose
spilling factor lies between two thresholds. Writing their positions in the format
`INTERACTIVE = .TRUE.` reads from stdin (bare blocks of direct coordinates, no header)
would complete the retraining workflow; check whether `structure[steps].to_POSCAR()`
can be reused for it.
