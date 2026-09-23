# The IRC coordinate can only be got by regex-parsing OUTCAR

`IBRION = 40` writes an intrinsic reaction coordinate, and py4vasp has no quantity for it.
The closest are `energy` and `structure`, neither of which carries the arc length.

So a workshop tutorial parsed the text output:

```python
re.compile(r"IRC \(A\):\s+([\-\d.Ee+]+)\s+E\(eV\):\s+([\-\d.Ee+]+)")
```

against a raw `OUTCAR` — the only place in that notebook touching a text file. One space
different in the format and the lists come back empty, and the next line raises an opaque
`IndexError` rather than saying the parse failed.

An IRC coordinate and its energy are a two-column trajectory, and the other trajectory
quantities (`energy`, `structure`, `force`) are already sliceable, so an `irc` quantity —
or at minimum exposure of the arc length alongside `energy[:]` — fits the existing shape
of the API.

Needs checking first whether VASP writes the IRC data to `vaspout.h5` at all, or only to
OUTCAR; that determines whether this is a reader or a request to VASP.
