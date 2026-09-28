# Selecting `current` on a dielectric function that has none returns NaN instead of raising

`DielectricFunctionHandler._init_components_dict` (`dielectric_function.py:174`) returns
`{None: 0, "density": 0, "current": 1}` unconditionally, but `_get_data` only builds a
component axis of length two when `_has_current_component()` is true. For an ionic
dielectric function it reshapes to `(1, 9, N, 2)`, so index 1 of that axis is out of
range for every real entry and `index.Selector`'s `np.average` reduction averages an
empty slice:

```python
calc.dielectric_function.read("current")["current"]   # array([nan, nan, ...])
calc.dielectric_function.plot("current")              # a line of NaN, drawn as nothing
```

numpy emits `RuntimeWarning: Mean of empty slice`, which nothing surfaces to the user.
`selections()` already reports the truth -- `components` is `["density"]` for this source
and `["density", "current"]` only when the current-current correlation is present -- so
the fix is to build the map from the same condition and let `index.Selector` raise its
usual `IncorrectUsage` for an unknown component, the way it already does for an unknown
direction.

Found while adding direction selection to `read`; that change did not introduce it and
does not widen it, because both `read` and `plot` share `_make_selector`.
