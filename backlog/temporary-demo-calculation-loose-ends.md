# Temporary demo calculations: a pickled copy does not own the directory

`demo.calculation()` without a path keeps its data in a temporary directory that is
removed once nothing uses the calculation. That is intended for example data, and
`Calculation.path()` as well as a quantity's `to_image`/`to_csv` with a relative filename
warn about it.

Low priority, because demo calculations are for doctests and trying py4vasp out, not for
production: a pickled copy does not own the directory. Once the original is collected,
every read of the copy raises `FileAccessError`, e.g. in multiprocessing workers. Copying
the data on unpickling or refusing to pickle would settle it.
