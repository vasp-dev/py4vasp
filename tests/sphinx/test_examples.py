# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import sys

import pytest

application = pytest.importorskip("sphinx.application")


@pytest.fixture(scope="module")
def sphinx_app(tmp_path_factory, not_core):
    if sys.version_info < (3, 12):
        pytest.skip("Sphinx example tests require Python 3.12 or higher.")
    tmp_path = tmp_path_factory.mktemp("sphinx")
    srcdir = "tests/sphinx/examples"
    confdir = "tests/sphinx/examples"
    outdir = tmp_path / "_build"
    doctreedir = tmp_path / "_doctree"
    app = application.Sphinx(
        srcdir=srcdir,
        confdir=confdir,
        outdir=outdir,
        doctreedir=doctreedir,
        buildername="hugo",
        status=None,
        warning=None,
        freshenv=True,
    )
    app.build(force_all=True)
    return app


def read_file_content(outdir, source_file):
    output_file = outdir / "hugo" / source_file
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert content.startswith("+++")
    return content


def test_convert_example_autodata(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example.md")
    expected_autodata_content = """\

{{< data name="Example" module="example" breadcrumbs="" >}}

{{< docstring >}}
An example class for demonstration purposes.

{{< /docstring >}}

{{< /data >}}

"""
    assert expected_autodata_content in content


def test_convert_example_autoclass_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example.md")
    expected_class_content = """\

{{< class name="Example" module="example" breadcrumbs="" >}}
{{< signature >}}
(*value*: `float`)
{{< /signature >}}

{{< docstring >}}
Bases: `object`

An example class for demonstration purposes.

{{< /docstring >}}

"""
    assert expected_class_content in content


def test_convert_example_init_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example.md")
    expected_private_method_content = """\

{{< method name="__init__" class="Example" module="example" breadcrumbs="" >}}
{{< signature >}}
(*value*: `float`)
{{< /signature >}}

{{< docstring >}}
Initialize the Example class with a value.

*   some list entry

*   some other list entry




#### Parameters



*value*: `float`
: <!---->
    The value to be stored in the instance.

{{< /docstring >}}

{{< /method >}}
"""
    assert expected_private_method_content in content


def test_convert_example_combined_returns_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example.md")
    expected_public_method_content = """\

{{< method name="combined_returns" class="Example" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *some_value*: `float`,
- *some_string*: `str | None` = ''

) → `tuple[float, str | None]`
{{< /signature >}}

{{< docstring >}}
Combine a float and a string in a tuple.



#### Parameters



*some_value*: `float`
: <!---->
    A value to be included in the tuple.

*some_string*: `str` = ''
: <!---->
    A string to be included in the tuple.

#### Returns


`tuple[float, str | None]`
: <!---->
    A tuple containing the float and a string representation.

"""
    assert expected_public_method_content in content


def test_convert_example_returns_type_without_desc_returns_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_return_types.md")
    expected_method_content = """\
{{< function name="returns_type_without_desc_returns" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *value1*: `float`,
- *value2*: `float | str`

) → `float | str`
{{< /signature >}}

{{< docstring >}}
Return value 2.



### Parameters



*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.

### Returns


`float | str`
: <!---->
    The second value.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_method_content in content


def test_convert_example_returns_type_without_returns_field_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_return_types.md")
    expected_method_content = """\

{{< function name="returns_type_without_returns_field" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *value1*: `float`,
- *value2*: `float | str`

) → `float | str`
{{< /signature >}}

{{< docstring >}}
Return value 2.



### Parameters



*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.

### Returns

`float | str`


{{< /docstring >}}

{{< /function >}}
"""
    assert expected_method_content in content


def test_convert_example_returns_type_without_returns_field_type_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_return_types.md")
    expected_method_content = """\
{{< function name="returns_type_without_returns_field_type" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *value1*: `float`,
- *value2*: `float | str`

) → `float | str`
{{< /signature >}}

{{< docstring >}}
Return value 2.



### Parameters



*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.

### Returns


`float | str`
: <!---->
    The second value.
    With another line!



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_method_content in content


def test_convert_example_returns_type_without_returns_field_desc_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_return_types.md")
    expected_method_content = """\
{{< function name="returns_type_without_returns_field_desc" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *value1*: `float`,
- *value2*: `float | str`

) → `float | str`
{{< /signature >}}

{{< docstring >}}
Return value 2.



### Parameters



*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.

### Returns

`float | str`


{{< /docstring >}}

{{< /function >}}
"""
    assert expected_method_content in content


def test_convert_example_params_types_only_in_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_param_types.md")
    expected_method_content = """\
{{< function name="params_types_only_in_signature" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *value1*: `float`,
- *value2*: `float | str` = 0

)
{{< /signature >}}

{{< docstring >}}
Example function with parameter types only in the signature.



### Parameters



*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str` = 0
: <!---->
    The second value, which can be a float or a string.
"""
    assert expected_method_content in content


def test_convert_example_params_types_only_in_field(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_param_types.md")
    expected_method_content = """\
{{< function name="params_types_only_in_field" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *value1*: `float`,
- *value2*: `float | str`

)
{{< /signature >}}

{{< docstring >}}
Example function with parameter types only in the field.



### Parameters



*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.
"""
    assert expected_method_content in content


def test_convert_example_params_types_in_signature_and_field(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_param_types.md")
    expected_method_content = """\
{{< function name="params_types_in_signature_and_field" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *value1*: `float`,
- *value2*: `float | str` = 0

)
{{< /signature >}}

{{< docstring >}}
Example function with parameter types mixed in both field and signature.



### Parameters



*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str` = 0
: <!---->
    The second value, which can be a float or a string.
"""
    assert expected_method_content in content


def test_convert_example_params_types_mismatched(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_param_types.md")
    expected_method_content = """\
{{< function name="params_types_mismatched" module="example" breadcrumbs="" >}}
{{< signature >}}
(
- *value1*: `float`,
- *value2*: `float | str` = 0

)
{{< /signature >}}

{{< docstring >}}
Example function with parameter types mismatched.



### Parameters



*value1*: `float | None`
: <!---->
    The first value.

*value2*: `float | str` = 0
: <!---->
    The second value, which can be a float or a string.
"""
    assert expected_method_content in content


def test_convert_example_dos_class(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_dos.md")
    expected_class_content = """
{{< class name="Dos" module="example_dos" breadcrumbs="" >}}
{{< signature >}}
(
- *data_context*,
- ***kwargs*

)
{{< /signature >}}

{{< docstring >}}
Bases: `Refinery`, `Mixin`

The density of states (DOS) describes the number of states per energy.

The DOS quantifies the distribution of electronic states within an energy range
in a material. It provides information about the number of electronic states at
each energy level and offers insights into the material’s electronic structure.
On-site projections near the atoms (projected DOS) offer a more detailed view.
This analysis breaks down the DOS contributions by atom, orbital and spin.
Investigating the projected DOS is often a useful step to understand the
electronic properties because it shows how different orbitals and elements
contribute and influence the material’s properties.

VASP writes the DOS after every calculation and the projected DOS if you set
[LORBIT](https://vasp.at/wiki/index.php/LORBIT) in the INCAR file. You can use this class to extract this data.
Typically you want to run a non self consistent calculation with a denser
mesh for a smoother DOS but the class will work independent of it. If you
generated a projected DOS, you can use this class to select which subset of
these orbitals to read or plot.


### Examples

If you want to visualize the total DOS, you can use the *plot* method. This will
show the different spin components if [ISPIN](https://vasp.at/wiki/index.php/ISPIN) = 2

~~~python
>>> calculation.dos.plot()
Graph(series=[Series(x=array(...), y=array(...), label='total', ...)],
    xlabel='Energy (eV)', ..., ylabel='DOS (1/eV)', ...)
~~~

If you need the raw data, you can read the DOS into a Python dictionary

~~~python
>>> calculation.dos.read()
{'energies': array(...), 'total': array(...), 'fermi_energy': ...}
~~~

These methods also accept selections for specific orbitals if you used VASP with
[LORBIT](https://vasp.at/wiki/index.php/LORBIT). You can get a list of the allowed choices with

~~~python
>>> calculation.dos.selections()
{'dos': ['default', 'kpoints_opt'], 'atom': [...], 'orbital': [...], 'spin': [...]}
~~~

"""
    assert expected_class_content in content


def test_convert_example_dos_selections(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_dos.md")
    expected_class_content = """
{{< method name="selections" class="Dos" module="example_dos" breadcrumbs="" >}}
{{< signature >}}
() → `dict`
{{< /signature >}}

{{< docstring >}}
Returns possible alternatives for this particular quantity VASP can produce.

The returned dictionary contains a single item with the name of the quantity
mapping to all possible selections. Each of these selection may be passed to
other functions of this quantity to select which output of VASP is used. Some
quantities provide additional elements which can be passed as selection for
other routines.



#### Returns


`dict`
: <!---->
    The key indicates this quantity and the values possible choices for arguments
    to other functions of this quantity.

"""
    assert expected_class_content in content


def test_convert_example_dos_to_dict(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_dos.md")
    expected_class_content = """
{{< method name="to_dict" class="Dos" module="example_dos" breadcrumbs="" >}}
{{< signature >}}
(*selection*: `str` = None) → `dict`
{{< /signature >}}

{{< docstring >}}
Read the DOS into a dictionary.

You will always get an “energies” component that describes the energy mesh for
the density of states. The energies are shifted with respect to VASP such that
the Fermi energy is at 0. py4vasp returns also the original “fermi_energy” so
you can revert this if you want. If [ISPIN](https://vasp.at/wiki/index.php/ISPIN) = 2, you will get the total
DOS spin resolved as “up” and “down” component. Otherwise, you will get just
the “total” DOS. When you set [LORBIT](https://vasp.at/wiki/index.php/LORBIT) in the INCAR file and pass in a
selection, you will obtain the projected DOS with a label corresponding to the
projection.



#### Parameters



*selection*: `str` = None
: <!---->
    A string specifying the projection of the orbitals. There are four distinct
    possibilities:

    *   To specify the **atom**, you can either use its element name (Si, Al, …)
        or its index as given in the input file (1, 2, …). For the latter
        option it is also possible to specify ranges (e.g. 1:4).

    *   To select a particular **orbital** you can give a string (s, px, dxz, …)
        or select multiple orbitals by their angular momentum (s, p, d, f).

    *   For the **spin**, you have the options up, down, or total.

    *   If you used a different **k**-point mesh choose “kpoints_opt” or “kpoints_wan”
        to select them instead of the default mesh specified in the KPOINTS file.


    You separate multiple selections by commas or whitespace and can nest them using
    parenthesis, e.g. *Sr(s, p)* or *s(up), p(down)*. The order of the selections
    does not matter, but it is case sensitive to distinguish p (angular momentum
    l = 1) from P (phosphorus).

    It is possible to add or subtract different components, e.g., a selection of
    “Ti(d) - O(p)” would project onto the d orbitals of Ti and the p orbitals of O and
    then compute the difference of these two selections.

    If you are unsure about the specific projections that are available, you can use

    ~~~python
    >>> calculation.projector.selections()
    {'atom': [...], 'orbital': [...], 'spin': [...]}
    ~~~

    to get a list of all available ones.



#### Returns


`dict`
: <!---->
    Contains the energies at which the DOS was evaluated aligned to the
    Fermi energy and the total DOS or the spin-resolved DOS for
    spin-polarized calculations. If available and a selection is passed,
    the orbital resolved DOS for the selected orbitals is included.




#### Examples

To obtain the total DOS along with the energy mesh and the Fermi energy you
do not need any arguments. For [ISPIN](https://vasp.at/wiki/index.php/ISPIN) = 2, this will “up” and “down”
DOS as two separate entries.

~~~python
>>> calculation.dos.to_dict()
{'energies': array(...), 'total': array(...), 'fermi_energy': ...}
~~~

Select the p orbitals of the first atom in the POSCAR file:

~~~python
>>> calculation.dos.to_dict(selection="1(p)")
{'energies': array(...), 'total': array(...), 'Sr_1_p': array(...),
    'fermi_energy': ...}
~~~

Select the d orbitals of Sr and Ti:

~~~python
>>> calculation.dos.to_dict("d(Sr, Ti)")
{'energies': array(...), 'total': array(...), 'Sr_d': array(...),
    'Ti_d': array(...), 'fermi_energy': ...}
~~~

Select the spin-up contribution of the first three atoms combined

~~~python
>>> calculation.dos.to_dict("up(1:3)")
{'energies': array(...), 'total': array(...), '1:3_up': array(...),
    'fermi_energy': ...}
~~~

Add the contribution of three d orbitals

~~~python
>>> calculation.dos.to_dict("dxy + dxz + dyz")
{'energies': array(...), 'total': array(...), 'dxy + dxz + dyz': array(...),
    'fermi_energy': ...}
~~~

Read the density of states generated by the ‘’’k’’’-point mesh in the KPOINTS_OPT
file

~~~python
>>> calculation.dos.to_dict("kpoints_opt")
{'energies': array(...), 'total': array(...), 'fermi_energy': ...}
~~~

"""
    assert expected_class_content in content


def test_convert_example_dos_to_graph(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_dos.md")
    expected_class_content = """
{{< method name="to_graph" class="Dos" module="example_dos" breadcrumbs="" >}}
{{< signature >}}
(*selection*: `str` = None) → `Graph`
{{< /signature >}}

{{< docstring >}}
Read the DOS and convert it into a graph.

The x axis is the energy mesh used in the calculation shifted such that the
Fermi energy is at 0. On the y axis, we show the DOS. For [ISPIN](https://vasp.at/wiki/index.php/ISPIN) = 2, the
different spin components are shown with opposite sign: “up” with a positive
sign and “down” with a negative one. If you used [LORBIT](https://vasp.at/wiki/index.php/LORBIT) in your VASP
calculation and you pass in a selection, py4vasp will add additional lines
corresponding to the selected projections.



#### Parameters



*selection*: `str` = None
: <!---->
    A string specifying the projection of the orbitals. There are four distinct
    possibilities:

    *   To specify the **atom**, you can either use its element name (Si, Al, …)
        or its index as given in the input file (1, 2, …). For the latter
        option it is also possible to specify ranges (e.g. 1:4).

    *   To select a particular **orbital** you can give a string (s, px, dxz, …)
        or select multiple orbitals by their angular momentum (s, p, d, f).

    *   For the **spin**, you have the options up, down, or total.

    *   If you used a different **k**-point mesh choose “kpoints_opt” or “kpoints_wan”
        to select them instead of the default mesh specified in the KPOINTS file.


    You separate multiple selections by commas or whitespace and can nest them using
    parenthesis, e.g. *Sr(s, p)* or *s(up), p(down)*. The order of the selections
    does not matter, but it is case sensitive to distinguish p (angular momentum
    l = 1) from P (phosphorus).

    It is possible to add or subtract different components, e.g., a selection of
    “Ti(d) - O(p)” would project onto the d orbitals of Ti and the p orbitals of O and
    then compute the difference of these two selections.

    If you are unsure about the specific projections that are available, you can use

    ~~~python
    >>> calculation.projector.selections()
    {'atom': [...], 'orbital': [...], 'spin': [...]}
    ~~~

    to get a list of all available ones.



#### Returns


`Graph`
: <!---->
    Graph containing the total DOS. If the calculation was spin polarized,
    the resulting DOS is spin resolved and the spin-down DOS is plotted
    towards negative values. If a selection is given the orbital-resolved
    DOS is given for the specified projectors.




#### Examples

For the total DOS, you do not need any arguments. py4vasp will automatically
use two separate lines, if you used [ISPIN](https://vasp.at/wiki/index.php/ISPIN) = 2 in the VASP calculation

~~~python
>>> calculation.dos.to_graph()
Graph(series=[Series(x=array(...), y=array(...), label='total', ...)],
    xlabel='Energy (eV)', ..., ylabel='DOS (1/eV)', ...)
~~~

Select the p orbitals of the first atom in the POSCAR file:

~~~python
>>> calculation.dos.to_graph(selection="1(p)")
Graph(series=[Series(..., label='total', ...), Series(..., label='Sr_1_p', ...)], ...)
~~~

Select the d orbitals of Sr and Ti:

~~~python
>>> calculation.dos.to_graph("d(Sr, Ti)")
Graph(series=[Series(..., label='total', ...), Series(..., label='Sr_d', ...),
    Series(..., label='Ti_d', ...)], ...)
~~~

Select the spin-up contribution of the first three atoms combined

~~~python
>>> calculation.dos.to_graph("up(1:3)")
Graph(series=[Series(..., label='total', ...), Series(..., label='1:3_up', ...)], ...)
~~~

Add the contribution of three d orbitals

~~~python
>>> calculation.dos.to_graph("dxy + dxz + dyz")
Graph(series=[Series(..., label='total', ...), Series(..., label='dxy + dxz + dyz', ...)], ...)
~~~

Read the density of states generated by the ‘’’k’’’-point mesh in the KPOINTS_OPT
file

~~~python
>>> calculation.dos.to_graph("kpoints_opt")
Graph(series=[Series(..., label='total', ...)], ...)
~~~

"""
    assert expected_class_content in content


def test_complicated_signatures_no_return_info(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content_no_return_info = """\
{{< function name="no_return_info" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `None`
{{< /signature >}}

{{< docstring >}}
Function without return type information in the docstring.



### Parameters



*a*: `int`
: <!---->
    An integer parameter.

*b*: `str`
: <!---->
    A string parameter.

### Returns

`None`


{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content_no_return_info in content


def test_complicated_signatures_no_return_description(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content_no_return_description = """\
{{< function name="no_return_description" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int`
{{< /signature >}}

{{< docstring >}}
Function with return type in signature but no Returns field.



### Returns

`int`


{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content_no_return_description in content


def test_complicated_signatures_return_with_description(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content_with_return_description = """\
{{< function name="return_with_description" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int`
{{< /signature >}}

{{< docstring >}}
Function with return type and description in Returns field.



### Returns


`int`
: <!---->
    The sum of the integer and the length of the string.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content_with_return_description in content


def test_complicated_signatures_return_with_type_only(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content_with_type_only = """\
{{< function name="return_with_type_only" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int`
{{< /signature >}}

{{< docstring >}}
Function with return type only in Returns field.



### Returns

`int`


{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content_with_type_only in content


def test_complicated_signatures_return_with_description_only(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content_with_description_only = """\
{{< function name="return_with_description_only" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int`
{{< /signature >}}

{{< docstring >}}
Function with return description only in Returns field.



### Returns


`int`
: <!---->
    The sum of the integer and the length of the string.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content_with_description_only in content


def test_complicated_signatures_return_with_no_info(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content_with_no_info = """\
{{< function name="return_with_no_info" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `-`
{{< /signature >}}

{{< docstring >}}
Function with no return type information.



### Parameters



*a*: `int`
: <!---->
    An integer parameter.

*b*: `str`
: <!---->
    A string parameter.

### Returns


`-`
: <!---->
    Returns a multiple of a.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content_with_no_info in content


def test_complicated_signatures_return_with_multiline_type_description_no_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    print(content)
    expected_content = """\
{{< function name="return_with_multiline_type_description_no_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `-`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and no type given.



### Returns


`-`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_field_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_field_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and type given in Returns field.



### Returns


`int`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_sig_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_sig_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and type given in signature.



### Returns


`int`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_tuple_sig_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_tuple_sig_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `Tuple[int, str]`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and Tuple type given in signature.



### Returns


`Tuple[int, str]`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_union_sig_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_union_sig_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int | str`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and Union type given in signature.



### Returns


`int | str`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_union2_sig_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_union2_sig_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int | str`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and Union2 type given in signature.



### Returns


`int | str`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_tuple_field_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_tuple_field_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `Tuple[int, str]`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and Tuple type given in Returns field.



### Returns


`Tuple[int, str]`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_union_field_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_union_field_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `Union[int, str]`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and Union type given in Returns field.



### Returns


`Union[int, str]`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_union2_field_type(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_union2_field_type" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int | str`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and Union2 type given in Returns field.



### Returns


`int | str`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_conflicting_type_01(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_conflicting_type_01" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and conflicting types.



### Returns


`int`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_conflicting_type_02(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_conflicting_type_02" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and conflicting types.



### Returns

`int`


{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content


def test_complicated_signatures_return_with_multiline_type_description_conflicting_type_03(
    sphinx_app,
):
    content = read_file_content(sphinx_app.outdir, "example_complicated_signatures.md")
    expected_content = """\
{{< function name="return_with_multiline_type_description_conflicting_type_03" module="example_complicated_signatures" breadcrumbs="" >}}
{{< signature >}}
(
- *a*: `int`,
- *b*: `str`

) → `int | str`
{{< /signature >}}

{{< docstring >}}
Function with type description across multiple lines and conflicting types.



### Returns


`int | str`
: <!---->
    This integer is returned
    and represents the sum of a
    and length of b.



{{< /docstring >}}

{{< /function >}}
"""
    assert expected_content in content
