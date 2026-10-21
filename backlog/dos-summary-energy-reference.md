# The DOS summary and `read()` use different energy references

`print(calculation.dos)` shows the energies as VASP wrote them, e.g.
`energies: [-7.00, 7.33]` for the demo data, while `dos.read()["energies"]` is shifted so
that the Fermi energy sits at 0 (`[-8.3, 6.03]`, with `fermi_energy` 1.3). The `read`
docstring says so; the summary does not say which reference it uses.

A user who integrates the DOS up to the Fermi energy from the summary's numbers mixes
the two references and gets a wrong electron count. A simulated user did exactly that
while trying the 1.0 beta and only recovered by reading the `read` docstring.

Either shift the summary the same way or label it, e.g.
`energies: [-7.00, 7.33] eV (unshifted, E_F = 1.30 eV)`. Check the other quantities with
a Fermi-energy shift (band, projected quantities) for the same mismatch.
