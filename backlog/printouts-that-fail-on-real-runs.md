# Printouts that fail on real runs although the data counts as available

Found while describing every routine of 89 real calculations with `backend.inspect`
(the same on master):

- `potential.print()` raises NoData on six gpu-lvhar runs whose potential counts as
  available.
- `local_moment.print()` raises TypeError on timing/elphon_fe_allinone.

Either the printout should cope with the data VASP wrote, or `is_available` should
report the printout as unavailable for it.
