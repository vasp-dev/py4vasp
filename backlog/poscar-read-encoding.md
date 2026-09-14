# Reading a structure file still depends on the locale

`py4vasp` now writes generated files as UTF-8, but `cli.py::_read_structure` reads
with `file.read_text()`, which uses the locale — cp1252 on Windows. A file py4vasp
wrote with a non-ASCII comment line therefore reads back mangled there, and the two
directions disagree.

Reading UTF-8 unconditionally would turn a legacy cp1252 POSCAR into a
`UnicodeDecodeError`, which escapes as a traceback. So: try UTF-8, fall back to the
locale, and convert a decode failure into `exception.IncorrectUsage` the way the
parser already reports a file that is not a POSCAR.
