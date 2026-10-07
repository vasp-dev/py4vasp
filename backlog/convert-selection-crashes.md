# `convert structure lammps -s …` crashes with a TypeError

Found by a simulated user who had only the documentation, while validating the change
that made `convert --help` name the one supported conversion.

`convert --help` advertises `-s/--selection` ("String to further clarify the specific
source of the quantity"), but any value crashes, `default` included:

    $ py4vasp convert structure lammps --from lr_run -s default
    TypeError: Structure.to_lammps() got an unexpected keyword argument 'selection'

The command exits with code 1 and prints a traceback with the source of `cli.py`, where
every other CLI error is a clean one-line message. `_convert_to_lammps` (`cli.py:76`)
passes `selection=` to `calculation.structure.to_lammps`, whose signature
(`structure.py:1287`) takes only `standard_form`. `test_convert_selection` mocks the
calculation, so the suite never sees the mismatch.

Either `Structure.to_lammps` accepts a selection like the other structure methods, so that
`-s final` converts the final structure, or `convert` drops the option. Whichever it is,
the CLI test should run against a real (demo) calculation rather than a mock, so that the
command and the method cannot drift apart again.
