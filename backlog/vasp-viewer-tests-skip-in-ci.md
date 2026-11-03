# The VASP Viewer round-trip tests never run in CI

`tests/third_party/view/test_vaspviewer.py` marks 16 tests `@hasVaspView`, which skips
them unless `vasp_viewer` is importable. None of the workflows in `.github/workflows/`
(`test_full`, `test_core`, `test_minimal`, `install`) installs the viewer, so every one
of them skips in CI. These are the only tests that feed py4vasp's payload through the
viewer's own validation (`View.to_vasp_viewer()`), so CI cannot see:

- a key py4vasp sends that the viewer rejects. When `IonArrow.preset` was added, the
  payload was rejected by vasp-studio `main` until its MR !312 landed (every
  `force`/`velocity`/`local_moment` plot raised `Rejection: ion_arrow_groups[0] has an
  unknown key 'preset'`), and only a local run against the viewer showed it;
- a value the viewer would store differently than py4vasp meant (e.g. colors, which the
  viewer lowercases to `#rrggbb`).

The config-level tests (`View.to_vasp_viewer_config()`) do run in CI, but they only
check what py4vasp sends, not whether the viewer accepts it.

It is blocked by the same thing as the `viewer` extra: `packages/py4vasp/pyproject.toml`
says to add `viewer = ["vasp-viewer"]` once the package is published to an index this
workspace can reach.

What would close it:
- once the viewer is reachable from CI, install it in at least `test_full`, so the
  `hasVaspView` tests run there;
- pin the viewer version CI uses and bump it when the payload contract changes, so a
  py4vasp change that needs a newer viewer shows up as a failing test, not after release.

Related: [vasp-viewer-is-undocumented] needs the same publication to give users an
install instruction.
