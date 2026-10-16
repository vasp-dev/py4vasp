# "magnetization" on a non-spin-polarized density is reported as a parse error

On a calculation without spin polarization, e.g. `demo.calculation()`,

```python
calculation.density.plot("magnetization")
```

raises `IncorrectUsage: Could not parse the selection "magnetization", please check the
spelling and capitalization`, which sends the user hunting for a typo. The equivalent
`calculation.density.plot("3")` gives the helpful `NoData: Density does not contain
magnetization. Please rerun VASP with ISPIN = 2 or LNONCOLLINEAR = T`.

The alias is only registered for collinear runs
(`DensityHandler._add_magnetization_for_charge_and_collinear`), so on other data the
selection is unknown rather than unavailable. Recognize "magnetization", "mag" and "m"
always, and raise the same `NoData` advice when the density has no magnetization. The
behavior is the same on `master`, so it predates the VASP Viewer work where it was found.
