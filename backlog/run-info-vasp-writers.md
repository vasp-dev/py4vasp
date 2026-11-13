# run_info: values VASP does not write yet

`run_info` reads its core values from `/setup`, `/metadata`, and `/messages`. Two values
still have to be inferred from results that VASP writes for a different purpose. A
dedicated dataset on the VASP side would make both reliable and cheap to read.

- **Number of ionic and electronic steps actually performed.** `num_ionic_steps` is
  the length of `intermediate/ion_dynamics/position_ions`. That is right for the runs
  checked so far, but it depends on VASP writing one position per step, and there is no
  count of electronic steps at all. A value in e.g. `metadata/performance` would also
  hold for killed runs.
- **The computed Fermi energy independently of the DOS.** `fermi_energy` comes from
  `results/electron_dos/efermi`. Check on the VASP side whether that group exists in
  every run; if not, the Fermi energy is missing exactly when no DOS was written.

Once VASP writes them, read the new datasets in `raw.RunInfo` and keep the current
source as fallback for older files, as `has_selective_dynamics` and the phonon counts do.
