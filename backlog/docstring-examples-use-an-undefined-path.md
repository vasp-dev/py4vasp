# Docstring examples use a `path` the reader never defined

Every calculation example starts with `demo.calculation(path)`. `tests/test_doctest.py`
injects `path` as a global, so the suite is green, but a user who pastes the example
into a notebook gets `NameError: name 'path' is not defined`. The README promises the
examples "can be copied and run as it stands".

Found by a simulated documentation-only user while validating the elastic-modulus
units change; it noticed on the first copy-paste. Every quantity is affected, not only
that one, so it wants one decision for all examples: write a literal such as
`demo.calculation("example")`, or say once, where the examples are introduced, that
`path` is any new directory.
