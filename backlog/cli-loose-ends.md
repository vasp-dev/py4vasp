# Command-line loose ends

Found by a simulated user trying the 1.0 beta from the documentation only:

- `py4vasp --version` answers "No such option". The usual way to check what is
  installed on the command line does not exist.
- `py4vasp generate kpath` rejects a calculation directory and needs the path to
  `vaspout.h5`, while `py4vasp convert ... -f` accepts the directory. The help says
  so and the error is clear, but the two subcommands should accept the same thing.
- `generate kpath` prints the Bravais lattice next to the space group, e.g.
  `I4/mmm (tI2)`. `tI2` reads like a Pearson symbol, which does not match a 7-atom cell;
  nothing in the documentation says what the label is.
