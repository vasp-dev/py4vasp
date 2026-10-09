# Temporary demo calculations: what `path()` does not cover

`demo.calculation()` without a path keeps its data in a temporary directory that is
removed once nothing uses the calculation. That is intended for example data, and
`Calculation.path()` now warns about it. Two ways around the warning are left:

- A quantity's `to_image(filename=...)` / `to_csv(filename=...)` resolve a relative
  filename against the calculation directory without calling `path()`
  (`_third_party/graph/mixin.py`), so `demo.calculation().dos.to_image(filename="dos.png")`
  saves into the temporary directory silently. Decide whether these warn as well.
- Low priority, because demo calculations are for doctests and trying py4vasp out, not
  for production: a pickled copy does not own the directory. Once the original is
  collected, every read of the copy raises `FileAccessError`, e.g. in multiprocessing
  workers. Copying the data on unpickling or refusing to pickle would settle it.
