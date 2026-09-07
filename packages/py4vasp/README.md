# py4vasp

This is a metadata-only distribution. It installs no Python modules of its own; it exists to
pull in [`py4vasp-core`](https://pypi.org/project/py4vasp-core) together with every optional
dependency, so that

~~~shell
pip install py4vasp
~~~

gives you a complete, ready-to-use installation. The code, the documentation and the issue
tracker all live with the main project:

- Documentation: https://vasp.at/py4vasp/latest
- Repository: https://github.com/vasp-dev/py4vasp
- Support forum: https://vasp.at/forum/

If you only need py4vasp's parsing capabilities and want to keep the dependency footprint
small, install `py4vasp-core` instead — it requires nothing but numpy and h5py, provides the
same `import py4vasp`, and enables the remaining features as soon as the corresponding
package is available.

Note that `pip uninstall py4vasp` removes only this metadata; use
`pip uninstall py4vasp py4vasp-core` to remove the code as well.
