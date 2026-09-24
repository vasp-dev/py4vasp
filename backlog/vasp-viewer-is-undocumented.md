# Nothing tells a user that the VASP Viewer exists, or how to get it

`View._ipython_display_` picks `vasp.viewer` over nglview when it is importable, and it
is the *only* backend that can animate a phonon mode — `to_ngl` raises
`NotImplemented` as soon as `View.phonon` is set (`_third_party/view/view.py:611`).

`grep -rin viewer docs/ README.md` returns **nothing**. The extras table in
`docs/_index.rst` lists `view → nglview, ase`, which is precisely the backend that
cannot show a phonon mode, so a user who reads it and installs `py4vasp[view]` is worse
off than one who reads nothing. `packages/py4vasp/pyproject.toml:36` reserves the extra
in a comment — *"Once vasp-viewer is published to an index this workspace can reach, add
viewer = ["vasp-viewer"]"* — so today there is no install instruction to give.

A simulated user who only read the documentation got as far as
`NotImplemented: Visualizing phonon modes is not available for NGLView` and had no
documented next step; they found the package only by running `pip show vasp_viewer`
(`0.0.1.dev459+g466e5858`, installed on that machine by someone else). Their verdict:
"the single most damaging gap".

This is larger than any one quantity — it affects every `to_view` and the whole viewer
integration — which is why it is recorded here rather than patched into one docstring.

What would close it:
- name the viewer in `docs/_index.rst`, next to the extras table, and say how to get it
  (or that it ships with VASP and is not on PyPI yet);
- add the `viewer` extra to `packages/py4vasp/pyproject.toml` once it is published;
- say in `docs/plot/view.rst` which backend can do what — isosurfaces, ion arrows and
  phonons each behave differently between nglview and the VASP Viewer.
