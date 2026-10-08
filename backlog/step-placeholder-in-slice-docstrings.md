# A literal `{step}` leaks into the docstrings of sliced quantities

`help(py4vasp.calculation.structure)` reads "You can also select specific {step}s or a
subset of {step}s as follows". The text comes from the shared template in
`src/py4vasp/_calculation/slice_.py`, which is formatted with a `step` name somewhere
but evidently not on every path that reaches the class docstring. Every quantity that
uses the template is probably affected, not only Structure.

Found by a simulated documentation-only user while validating `demo.calculation()`
without a path.
