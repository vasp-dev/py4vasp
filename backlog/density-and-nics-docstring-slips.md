# Docstring slips in Density and Nics

A simulated user who only read the documentation stumbled over three small errors:

- The class docstrings of `Nics` (`nics.py:250`) and `Density` (`density.py:324`) say
  "you can read the **band** data into a Python dictionary", copied from `Band`.
- The `Nics.to_contour` examples (`nics.py:505` and `:510`) pass `selection=0.3`, while
  the text around them speaks of the "xy" selection and of the isotropic shift. Copied
  as is, they raise `IncorrectUsage: Selection must be a string`.
- The examples of `Density.to_view` and `Nics.to_contour` start from
  `Calculation.from_path(".")`, while the rest of the documentation uses
  `demo.calculation()` so that every example runs as pasted.

Each is a one-line fix; check the neighbouring examples of the same methods for the
same slips while at it.
