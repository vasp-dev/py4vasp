# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import dataclasses
import types

import numpy as np
import pytest

from py4vasp import _config, exception
from py4vasp._calculation.nics import Nics
from py4vasp._calculation.structure import Structure
from py4vasp._third_party import view


@pytest.fixture
def nics_on_a_grid(raw_data):
    raw_nics = raw_data.nics("on-a-grid")
    nics = Nics.from_data(raw_nics)
    nics.ref = types.SimpleNamespace()
    transposed_nics = np.array(raw_nics.nics_grid).T
    nics.ref.structure = Structure.from_data(raw_nics.structure)
    nics.ref.output = {
        "method": "grid",
        "nics": transposed_nics.reshape((10, 12, 14, 3, 3)),
    }
    return nics


@pytest.fixture
def nics_at_points(raw_data):
    raw_nics = raw_data.nics("at-points")
    nics = Nics.from_data(raw_nics)
    nics.ref = types.SimpleNamespace()
    nics.ref.structure = Structure.from_data(raw_nics.structure)
    nics.ref.output = {
        "method": "positions",
        "nics": raw_nics.nics_points,
        "positions": np.transpose(raw_nics.positions),
    }
    return nics


@dataclasses.dataclass
class Normal:
    normal: str
    expected_rotation: np.ndarray


@pytest.fixture(
    params=[
        Normal(normal="auto", expected_rotation=np.eye(2)),
        Normal(normal="x", expected_rotation=np.array([[0, -1], [1, 0]])),
        Normal(normal="y", expected_rotation=np.diag((1, -1))),
        Normal(normal="z", expected_rotation=np.eye(2)),
    ]
)
def normal_vector(request):
    return request.param


@pytest.fixture(
    params=[
        None,
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
        "xx + yy",
        "xx yy",
        "isotropic",
    ],
)
def selection(request):
    return request.param


def test_read_grid(nics_on_a_grid, Assert):
    actual = nics_on_a_grid.read()
    actual_structure = actual.pop("structure")
    Assert.same_structure(actual_structure, nics_on_a_grid.ref.structure.read())
    assert actual.keys() == nics_on_a_grid.ref.output.keys()
    assert actual["method"] == nics_on_a_grid.ref.output["method"]
    Assert.allclose(actual["nics"], nics_on_a_grid.ref.output["nics"])


def test_read_points(nics_at_points, Assert):
    actual = nics_at_points.read()
    actual_structure = actual.pop("structure")
    Assert.same_structure(actual_structure, nics_at_points.ref.structure.read())
    assert actual.keys() == nics_at_points.ref.output.keys()
    assert actual["method"] == nics_at_points.ref.output["method"]
    Assert.allclose(actual["nics"], nics_at_points.ref.output["nics"])
    Assert.allclose(actual["positions"], nics_at_points.ref.output["positions"])


def get_3d_tensor_element_from_grid(tensor, element: str):
    if element == "3x3":
        return tensor
    if element == "xx":
        return tensor[..., 0, 0]
    elif element == "xy":
        return tensor[..., 0, 1]
    elif element == "xz":
        return tensor[..., 0, 2]
    elif element == "yx":
        return tensor[..., 1, 0]
    elif element == "yy":
        return tensor[..., 1, 1]
    elif element == "yz":
        return tensor[..., 1, 2]
    elif element == "zx":
        return tensor[..., 2, 0]
    elif element == "zy":
        return tensor[..., 2, 1]
    elif element == "zz":
        return tensor[..., 2, 2]
    elif element == "xx + yy":
        return tensor[..., 0, 0] + tensor[..., 1, 1]
    elif element == "xx yy":
        return [tensor[..., 0, 0], tensor[..., 1, 1]]
    elif element in [None, "isotropic"]:
        return (tensor[..., 0, 0] + tensor[..., 1, 1] + tensor[..., 2, 2]) / 3.0
    else:
        raise ValueError(
            f"Element {element} is unknown by get_3d_tensor_element_from_grid."
        )


def test_plot(nics_on_a_grid, selection, Assert):
    tensor = nics_on_a_grid.ref.output["nics"]
    element = get_3d_tensor_element_from_grid(tensor, selection)
    structure_view = nics_on_a_grid.plot(selection)
    expected_view = nics_on_a_grid.ref.structure.plot()
    Assert.same_structure_view(structure_view, expected_view)
    if not (isinstance(element, list)):
        element = [element]
        selection_list = [selection]
    else:
        selection_list = str.split(selection)
    assert len(structure_view.grid_scalars) == len(element)
    for grid_scalar, e, s in zip(structure_view.grid_scalars, element, selection_list):
        assert grid_scalar.label == (f"{s} NICS" if s else "isotropic NICS")
        assert grid_scalar.quantity.ndim == 4
        Assert.allclose(grid_scalar.quantity, e)
        assert len(grid_scalar.isosurfaces) == 2
        assert grid_scalar.isosurfaces == [
            view.Isosurface(1.0, _config.VASP_COLORS["blue"], 0.6),
            view.Isosurface(-1.0, _config.VASP_COLORS["red"], 0.6),
        ]


@pytest.mark.parametrize("supercell", (2, (3, 1, 2)))
def test_plot_supercell(nics_on_a_grid, supercell, Assert):
    view = nics_on_a_grid.plot(supercell=supercell)
    Assert.allclose(view.supercell, supercell)


def test_plot_user_options(nics_on_a_grid):
    view = nics_on_a_grid.plot(isolevel=0.9, opacity=0.2)
    assert len(view.grid_scalars) == 1
    grid_scalar = view.grid_scalars[0]
    assert len(grid_scalar.isosurfaces) == 2
    for idx, isosurface in enumerate(grid_scalar.isosurfaces):
        assert isosurface.isolevel == (-1.0) ** (idx) * 0.9
        assert isosurface.opacity == 0.2


@pytest.mark.parametrize(
    "kwargs, index, position",
    (({"a": 0.1}, 0, 1), ({"b": 0.7}, 1, 8), ({"c": 1.3}, 2, 4)),
)
def test_to_contour(nics_on_a_grid, kwargs, index, position, Assert, selection):
    graph = nics_on_a_grid.to_contour(selection=selection, **kwargs)
    slice_ = [slice(None), slice(None), slice(None)]
    slice_[index] = position
    tensor = nics_on_a_grid.ref.output["nics"]
    scalar_data = get_3d_tensor_element_from_grid(tensor, selection)
    if not (isinstance(scalar_data, list)):
        scalar_data = [scalar_data[tuple(slice_)]]
        selection_list = [selection]
    else:
        scalar_data = [s[tuple(slice_)] for s in scalar_data]
        selection_list = str.split(selection)
    assert len(graph) == len(scalar_data)
    for series, e, s in zip(graph, scalar_data, selection_list):
        assert series.label == (
            f"{s if s else 'isotropic'} NICS contour ({list(kwargs.keys())[0]})"
        )
        Assert.allclose(series.data, e)


def test_to_contour_supercell(nics_on_a_grid, Assert):
    graph = nics_on_a_grid.to_contour(b=0, supercell=2)
    Assert.allclose(graph.series[0].supercell, (2, 2))
    graph = nics_on_a_grid.to_contour(b=0, supercell=(2, 1))
    Assert.allclose(graph.series[0].supercell, (2, 1))


def test_to_contour_normal(nics_on_a_grid, normal_vector, Assert):
    graph = nics_on_a_grid.to_contour(c=0.5, normal=normal_vector.normal)
    rotation = normal_vector.expected_rotation
    lattice_vectors = nics_on_a_grid.ref.structure.lattice_vectors()
    expected_lattice = lattice_vectors[:2, :2] @ rotation
    Assert.allclose(graph.series[0].lattice.vectors, expected_lattice)


def test_to_numpy_grid(nics_on_a_grid, selection, Assert):
    tensor = nics_on_a_grid.ref.output["nics"]
    element = get_3d_tensor_element_from_grid(tensor, selection or "3x3")
    Assert.allclose(nics_on_a_grid.to_numpy(selection), element)


def test_to_numpy_points(nics_at_points, selection, Assert):
    tensor = nics_at_points.ref.output["nics"]
    element = get_3d_tensor_element_from_grid(tensor, selection or "3x3")
    Assert.allclose(nics_at_points.to_numpy(selection), element)


def test_nics_with_points_not_with_plotting_routines(nics_at_points):
    with pytest.raises(exception.IncorrectUsage):
        nics_at_points.plot()
    with pytest.raises(exception.IncorrectUsage):
        nics_at_points.to_contour(a=0)


def test_print_grid(nics_on_a_grid, format_):
    actual, _ = format_(nics_on_a_grid)
    expected_text = """\
nucleus-independent chemical shift:
    structure: Sr2TiO4
    grid: 10, 12, 14
    tensor shape: 3x3"""
    assert actual == {"text/plain": expected_text}


def test_print_points(nics_at_points, format_):
    actual, _ = format_(nics_at_points)
    # print(nics_at_points.ref.output["nics"].shape)
    # print(nics_at_points.ref.output["positions"].shape)
    # print(actual["text/plain"])
    expected_text = """\
nucleus-independent chemical shift:
    structure: Fe3O4
    NICS at  -7.304104   1.449073   0.648033: |
        +4.694641e+00   -5.183351e+00   -1.427897e+01
        +5.200166e+00   -1.175530e+01   +3.814939e+00
        +2.430136e-01   -1.079061e+01   -2.025810e+01

    NICS at  -5.002188   0.626277   2.613072: |
        +4.563070e-01   -2.050131e+01   +6.691904e+00
        -5.004095e+00   -1.433975e+01   -1.286719e+01
        +3.159335e+00   +1.189612e+01   -1.092822e+00

    NICS at -18.696392 -10.011902 -16.764380: |
        -3.068827e+00   -4.804258e+00   -1.008576e+01
        +2.345812e-01   -6.159534e+00   -6.857905e+00
        -1.470932e+01   +5.895868e+00   +9.542238e+00

    NICS at   3.007405 -21.113625  -0.092836: |
        +1.078071e+01   -1.064699e+01   -5.429551e+00
        +1.428687e+01   -7.284867e+00   +4.199830e+00
        +4.023377e+00   -1.501387e+01   +3.156173e+00

    NICS at -18.094453 -12.962680   0.069922: |
        -2.615400e+00   +1.810240e+01   +2.789908e+00
        +0.000000e+00   -3.706290e-01   +1.705365e+01
        +1.735478e+01   +1.334732e+00   -3.634052e+00

    NICS at  -7.361122   2.926664  -0.696102: |
        +1.357860e+01   +2.362590e+00   +1.848318e+00
        +1.848903e+00   +9.305337e+00   -1.299491e+01
        -7.939417e+00   -9.494421e+00   +6.613463e+00

    NICS at  14.634796  -3.207906  19.547557: |
        +1.979859e+01   -2.653274e+00   +3.156791e+00
        +2.091653e+01   +1.091922e+01   -1.171888e+01
        +4.900611e+00   +6.873776e+00   +7.259524e+00

    NICS at  -7.846493  -0.263113   5.585974: |
        -8.470786e+00   +4.571152e+00   -8.230762e+00
        -3.590378e+00   -1.410210e+00   -1.845829e+01
        -7.969189e+00   +2.574889e-01   -1.331166e+01

    NICS at  -1.825897  -3.264518 -24.374443: |
        -3.442170e+00   +1.688988e+00   -6.381746e+00
        +2.976289e+00   +9.061992e+00   -1.959381e+01
        +1.881344e+01   -1.566144e+01   -4.018061e+00

    NICS at  11.902162  -5.439370  -2.319759: |
        -2.699602e+00   -3.821365e+00   -0.000000e+00
        +1.130222e+01   -1.409428e+01   -8.729385e+00
        -9.763199e+00   -1.202037e+01   +3.395485e+00

    NICS at  -6.959072  -0.621185   0.322279: |
        +2.470628e+00   +2.959955e+00   -2.044962e+00
        +7.242086e+00   +2.729503e+00   +2.721577e+00
        -2.331702e+00   +2.761745e+01   +1.912736e+01

    NICS at  15.015083  -6.751288   6.890847: |
        -7.988590e+00   -6.367894e+00   +1.242064e+00
        -8.748897e+00   -4.022748e+00   -2.468031e+00
        +8.837582e+00   +1.000000e-14   -6.497720e+00

    NICS at  16.894270  -0.300547 -11.266543: |
        +1.656996e+00   +7.360779e-02   +1.584841e+01
        +7.481183e+00   +1.437385e+01   -2.323096e-01
        -2.688914e+01   +4.787002e+00   -1.145152e+01

    NICS at   3.555234  -1.046950  -4.440194: |
        -4.985460e+00   -3.648775e+00   -3.622751e+00
        +2.255372e+01   +2.895100e+00   -2.368240e+00
        -1.141582e+01   +7.770568e-01   -3.260583e+00

    NICS at   6.957972  -9.965086  -7.092831: |
        +3.617409e+00   +2.184376e+00   +3.902797e+00
        -7.225056e+00   -4.481374e-01   +9.450571e+00
        +6.696913e+00   -3.316046e+00   -1.840920e+01

    NICS at   4.294869 -19.226175 -15.410181: |
        -7.560594e-01   +5.454260e+00   +5.681036e+00
        -6.058516e+00   +1.404451e+01   +3.724793e-01
        +9.436764e+00   -5.775561e-01   -3.868882e+00

    NICS at   4.231413   4.695152 -12.501260: |
        +1.996379e+01   +1.335545e+01   -4.662945e+00
        -1.336483e+00   -8.315209e+00   +2.530856e+00
        +3.059723e+00   -2.731653e+00   -7.622886e+00

    NICS at -11.521913  -5.186734  -1.625000: |
        +6.493858e+00   -5.996264e+00   +2.992277e+01
        +2.433043e+00   +1.607604e+01   +1.419661e+01
        +9.412906e+00   +1.741073e+01   -6.855529e+00

    NICS at  -9.618304  -6.699152  -6.544972: |
        +1.746717e+01   -1.426389e+00   -1.538678e+01
        +2.020015e+01   +6.612918e+00   +1.093272e+00
        +1.025484e+01   +9.673384e+00   +1.228684e+01

    NICS at   0.165904   6.576176  -3.085082: |
        +9.712597e+00   +2.142417e+01   +2.897905e+00
        -3.920739e+00   -1.284795e+01   -1.390418e+01
        -9.161228e-01   -2.757164e+01   -3.687364e+00

    NICS at  -9.634815  -3.616151   8.850635: |
        -1.175423e+01   -1.368415e+00   -9.504002e-01
        +2.042675e+00   -1.140530e+00   +7.058434e+00
        +5.193849e+00   -5.891279e+00   +3.175371e+00

    NICS at   3.426302 -17.765094  -1.363607: |
        +1.267784e+01   -8.872187e+00   +2.234347e+01
        -7.769600e+00   +3.957752e+00   +2.850152e+00
        -7.526571e+00   +2.584312e+01   +6.184512e+00

    NICS at   2.064369   7.718566  -2.768847: |
        -5.909842e+00   +7.971057e+00   -1.080536e+01
        -8.735288e+00   -1.589747e+00   +4.927167e-01
        -1.240443e+01   -1.534917e+01   +1.683888e+01

    NICS at   7.023428  -9.361406   6.328781: |
        +8.970133e-01   +1.203956e+01   -4.152281e+00
        +1.756342e+01   -1.009195e+01   -7.275412e+00
        +2.330293e+01   +2.934394e+00   +4.886424e+00

    NICS at  -0.487842  -3.128737 -11.666396: |
        +5.351604e-01   -9.434268e+00   +1.263728e+01
        +1.703485e+01   -2.031534e+00   -5.064765e+00
        -7.239495e+00   -2.233703e+00   +1.220220e+01

    NICS at  17.469952 -24.154747  -7.900294: |
        -1.001938e+01   +2.327711e+01   +1.235977e+01
        +8.130745e+00   -1.319092e+01   -2.459539e+00
        +1.006923e+01   -7.685305e+00   -1.012611e+01

    NICS at   3.042242 -18.882347   3.358723: |
        +4.282440e+00   -1.560029e+01   -5.186540e+00
        +1.073279e+01   +5.634687e+00   -4.452733e+00
        +6.992725e+00   -1.144411e+01   +1.510358e+01

    NICS at  20.600709   6.124298 -12.613795: |
        +9.565343e+00   +1.930580e+00   +2.054922e+00
        +1.838346e+01   +7.483207e-01   -1.973250e+01
        -1.676990e+00   +7.339143e-01   +7.325277e+00

    NICS at  -1.755680   1.198420 -20.685987: |
        +2.761839e+00   -2.187069e+01   -6.692061e-01
        +1.614669e+01   +9.199939e+00   -7.797973e+00
        -3.963727e+00   -1.351783e-01   +5.013144e+00

    NICS at   5.755464  -6.268455   9.524117: |
        +1.486908e+00   -4.051026e+00   +5.712882e+00
        -1.029706e+01   +7.565795e+00   +6.457649e-01
        -7.191738e+00   +1.272648e+01   +1.411854e+01

    NICS at   5.838207  -1.417218  -0.419066: |
        -2.103179e+01   -2.473423e-01   -1.055087e+01
        -1.186670e+01   -7.758678e+00   +8.168070e+00
        +1.107224e+01   -1.148217e+01   -1.525291e+01

    NICS at  -9.256858  -3.947682  -8.826087: |
        +3.772620e+00   +5.375653e+00   +1.631487e+01
        -8.113157e+00   -1.282585e+01   +1.549365e-01
        -1.354913e+00   +1.131604e+00   +4.194446e+00

    NICS at -13.394725  12.876220 -13.923402: |
        +1.887645e+01   +8.930812e+00   -1.258672e+01
        +3.962532e+00   -1.196826e+01   +4.756456e+00
        -9.992477e+00   -5.107931e+00   -7.595831e+00

    NICS at -14.011332  25.933344  -2.709623: |
        -1.269405e+00   -3.181856e+00   -6.776799e+00
        -5.645493e+00   -8.884116e+00   +1.163717e+01
        +1.704871e+01   -1.998640e+01   +5.912235e-01

    NICS at   5.080246   0.093116 -11.300151: |
        -4.221880e+00   -2.246413e+00   +1.696069e+01
        +3.779179e+00   -7.361045e+00   -1.081943e+01
        -9.220361e+00   -1.201589e+01   +5.456585e+00

    NICS at   8.347539 -11.193757   6.756724: |
        +4.724698e+00   -3.645963e-02   -3.113905e+00
        -4.615696e+00   -1.077819e+01   -7.630202e+00
        +8.665499e-01   +1.476779e+01   -1.245059e+01

    NICS at  -6.617393   0.685954   2.422823: |
        +1.262310e+01   +7.784303e+00   +3.528983e+00
        -8.943885e+00   -1.135055e+01   -4.374221e+00
        -5.640797e+00   +1.469019e+01   -6.863647e+00

    NICS at  -9.554431   9.138379  -3.824988: |
        -2.362679e+00   +1.109351e+01   -1.820302e+00
        -1.826453e+01   +1.326112e+01   +5.338517e+00
        -1.022613e+01   +1.373733e+01   -5.800531e+00

    NICS at  19.974824   8.158172   5.480654: |
        +7.066245e+00   +1.351477e+01   -7.951652e+00
        +7.151928e+00   +1.349271e+01   +1.189279e+01
        -4.270435e+00   -2.126055e+00   -1.124728e+01

    NICS at   2.507208   3.932324 -11.651205: |
        +3.839297e+00   +1.286777e+00   +1.169743e+01
        +3.237935e+00   +7.769755e-01   -1.104314e+01
        -1.151985e+01   +2.518183e-01   -1.330808e+01

    NICS at  -5.279637   0.945834  -1.126406: |
        -1.244218e+01   +8.521313e+00   -7.564949e-01
        +1.554707e+01   -1.238513e+01   +5.339286e+00
        +5.565198e+00   -5.240077e+00   +1.462500e+01

    NICS at   2.225782  -1.597268  -5.656815: |
        +1.665867e+01   +1.233493e+01   -4.227795e+00
        -5.302657e+00   +5.641368e-01   +1.726891e+01
        +1.862386e+00   -1.263760e+01   -6.020215e+00

    NICS at  -8.429268 -12.309876 -11.493022: |
        +1.432544e+01   -1.979814e+01   +9.460629e-02
        +1.307647e+01   -6.750736e+00   +1.835814e+00
        -5.071310e+00   -1.407541e+01   +7.644144e+00

    NICS at  -7.734288  -7.553198 -10.735938: |
        -1.864777e+01   +4.806809e-01   +9.322131e+00
        +1.497883e+01   -1.733471e+01   -2.098157e+01
        +4.452785e+00   +1.195396e+01   +1.031631e+01

    NICS at  -6.278696   5.906804  -1.297238: |
        -3.291433e+00   +6.910588e+00   -1.374616e+01
        +4.251582e+00   -9.793002e-01   +1.127995e+01
        +4.009054e+00   +1.514217e-01   +3.505019e-02

    NICS at   8.881675  -2.901932   4.944038: |
        +1.304960e+01   -1.190701e+00   -2.402231e-01
        -1.184187e+01   +5.946074e+00   +1.632023e+01
        -1.637909e+01   +2.370412e+01   -1.039177e+01

    NICS at   1.307732  -4.447010   5.552648: |
        +1.556156e+01   -3.310096e+00   -5.548938e+00
        -6.549523e+00   +8.978247e-01   -1.182765e+01
        -8.840326e+00   +1.381975e+01   +5.132032e+00

    NICS at   5.336030  -9.215581  -0.105443: |
        -5.442472e+00   -7.889355e+00   +5.801188e+00
        -1.413712e+01   -1.673876e+01   +2.522653e+00
        -6.770997e+00   -4.443456e+00   +2.366101e+00

    NICS at -10.053176   3.611022   2.028683: |
        -1.463496e+01   +6.032809e+00   +1.562225e+01
        +6.921948e+00   +7.579664e-01   +1.952505e+01
        -8.877000e+00   -4.893912e+00   +1.354426e+01

    NICS at  -8.911340 -10.687320  -8.460478: |
        +2.254708e+00   +1.776938e+01   +5.037829e-01
        +1.394437e+01   +3.971275e+00   -1.125594e+00
        +1.320974e+01   -1.831827e-01   +5.629069e+00"""
    assert actual == {"text/plain": expected_text}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.nics("on-a-grid")
    check_factory_methods(Nics, data)
