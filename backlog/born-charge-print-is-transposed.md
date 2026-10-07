# `born_effective_charge.print()` shows every tensor transposed relative to OUTCAR

`BornEffectiveChargeHandler.__str__` (`born_effective_charge.py`) imitates the OUTCAR
block `BORN EFFECTIVE CHARGES (including local field effects)` and prints
`charge_tensor[0]`, `[1]`, `[2]` as the rows labelled 1, 2, 3.

VASP writes `BORN(IDIR,:,N)` per row, so in OUTCAR the row label is the **field**
direction (`linear_response.F:970-974`, `pead.F:2370-2375`). The same Fortran array is
written to `vaspout.h5` without a transpose (`vhdf5.F:1568-1587`), so the numpy array
py4vasp reads has the **displacement** direction as axis 1 and the field as axis 2.
Printing `charge_tensor[i]` as row `i` therefore shows the transpose of what OUTCAR
shows under the identical header.

Nothing is wrong for diagonal tensors, which is why it went unnoticed. For a
low-symmetry polar material a user comparing py4vasp's output to OUTCAR, or copying the
printed numbers into `PHON_BORN_CHARGES`, gets the transpose. `to_INCAR` already writes
the correct orientation, and the tests in `test_born_effective_charge.py` document the
VASP convention.

Fix: print `charge_tensor.T` row by row, and update `REFERENCE_OUTPUT` in the test,
whose `np.arange` data is not symmetric and will pin the change. Decide at the same time
whether `read()` should keep the raw orientation; changing it would break users who
already transpose themselves, so documenting the axis order in its docstring may be
the safer half.
