# Missing data in the example data is blamed on the INCAR

`demo.calculation(selection="metal").dielectric_function.read()` raises `NoData` with
the advice to check that "the INCAR tags of the calculation produce this data, that VASP
finished, and that it did not exit with an error". The example data never had an INCAR
or a VASP run; the selection simply does not contain that quantity.

The help of `demo.calculation` says so, and `selections(only_available=True)` lists
what is there, but the error is what the user reads. When the data comes from the demo,
the message should name the quantities the selection contains or suggest the default
selection. Related to `missing-data-advice-does-not-name-the-incar-tags.md`.

Found by a simulated documentation-only user while validating `demo.calculation()`
without a path.
