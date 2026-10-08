# `import py4vasp.calculation` fails

`py4vasp.calculation` is a `Calculation` object, not a module, so the import a Python
user tries first,

    import py4vasp.calculation
    from py4vasp.calculation import dos

raises `ModuleNotFoundError`. The documentation writes it as `py4vasp.calculation.dos`
everywhere, which reads like a module path, and the cross references point there too.

The simulated user of #354 hit this while looking for `calculation.mass`, which is
reachable only through the link in the `masses` parameter text. A module
`py4vasp/calculation.py` that forwards attribute access to the default `Calculation`
(PEP 562 `__getattr__`) would make both spellings work; decide whether
`from py4vasp.calculation import dos` should then give the quantity bound to the
current directory, which is what the attribute access gives today.
