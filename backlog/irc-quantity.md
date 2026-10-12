# The IRC arc length can only be got by regex-parsing OUTCAR

`IBRION = 40` follows the intrinsic reaction coordinate, and `calculation.reaction_path`
now maps such a run onto distances between pairs of atoms. The energies along the run
are in `energy[:]`. What is still missing is the arc length along the mass-weighted
path that VASP prints as

```
    IRC (A):   0.00025036 E(eV): -.51954367E+02
```

It is only written to OUTCAR, not to `vaspout.h5`, so a workshop tutorial parsed it
with a regular expression. One space different in the format and the lists come back
empty, and the next line raises an opaque `IndexError` rather than saying the parse
failed.

This is a request to VASP first: write the arc length of every step to `vaspout.h5`.
Then expose it in py4vasp, e.g. as the x axis of `reaction_path.to_path` instead of the
step index, or alongside `energy[:]`, so an energy profile along the IRC needs no text
parsing.
