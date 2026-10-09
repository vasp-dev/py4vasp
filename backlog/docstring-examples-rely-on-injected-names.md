# Docstring examples use names the reader never imported

The examples no longer use an undefined `path`, but they still lean on names a user who
copies them does not have. `tests/test_doctest.py` injects `np` and `py4vasp` as
globals, and doctest runs every example with the globals of the module it is defined
in. So the suite is green while a fresh notebook raises a `NameError`:

- `Graph.to_csv` and the other `graph.py` examples call `np.array` and `Graph(...)`.
- The `Contour` examples use `np`, `Lattice`, `Contour` and `Graph`.
- Several calculation examples, e.g. the `Calculation` class and `from_archive`, call
  `py4vasp.demo.calculation()` without `import py4vasp`.

The fix has the same shape as the one for `path`: add the imports to the examples and
let `test_no_example_reads_an_undefined_path` grow into a check that every name an
example loads is assigned or imported earlier in it (or is a builtin). Running the
examples with empty globals instead of the module's would make the doctests enforce the
same thing.
