# Slow-growth and blue-moon results are only in REPORT

A slow-growth simulation (`IBRION = 0`, an `IS` or other constrained coordinate in
ICONST, `INCREM`, `LBLUEOUT = T`) writes the value of the constrained coordinate (`cc`)
and the free-energy gradient (`b_m`) of every step to REPORT only. The transition-state
tutorial (part 4, exercise 14) extracts them with

```shell
grep cc REPORT | awk '{print $3}' > xxx
grep b_m REPORT | awk '{print $2}' > fff
```

and integrates the gradient with a separate script to obtain the free-energy profile
ΔA(ξ) and the barrier.

This is a request to VASP first: write the constrained coordinates and the blue-moon
gradient of every step to `vaspout.h5`. Then py4vasp could offer the gradient and its
integral along the coordinate as a quantity, with a `plot` of the free-energy profile,
replacing the grep pipeline and the integration script. `calculation.reaction_path`
already prepares the IRCCAR and ICONST files that such a run starts from.
