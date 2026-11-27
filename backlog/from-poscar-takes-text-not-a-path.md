# `Structure.from_POSCAR` takes the text of a file, and says so nowhere

A user who calls `calculation.structure.from_POSCAR("POSCAR")` with a file name gets
"The POSCAR ended before all required parts were read ... Please check that the file is
complete", although the file is complete: the method parses its argument as the
contents of a POSCAR. Its docstring documents only `elements`, not `poscar`. An error
that recognizes a path ("this looks like a file name; pass its text, e.g.
`Path(name).read_text()`"), or accepting a path, would save the detour.

Also: the repr of such a structure claims `Structure.from_path(PosixPath('<cwd>'))`,
although nothing was read from the current directory, and `repr(structure[0])` equals
`repr(structure)`, so the repr hides the selected step.
