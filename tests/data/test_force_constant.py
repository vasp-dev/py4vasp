import pytest
import types
from py4vasp.data import ForceConstant, Structure


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_force_constants = raw_data.force_constant("Sr2TiO4")
    force_constants = ForceConstant(raw_force_constants)
    force_constants.ref = types.SimpleNamespace()
    force_constants.ref.structure = Structure(raw_force_constants.structure)
    force_constants.ref.force_constants = raw_force_constants.force_constants
    return force_constants


def test_Sr2TiO4_read(Sr2TiO4, Assert):
    actual = Sr2TiO4.read()
    reference_structure = Sr2TiO4.ref.structure.read()
    for key in actual["structure"]:
        if key in ("elements", "names"):
            assert actual["structure"][key] == reference_structure[key]
        else:
            Assert.allclose(actual["structure"][key], reference_structure[key])
    Assert.allclose(actual["force_constants"], Sr2TiO4.ref.force_constants)


def test_Sr2TiO4_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    reference = """
Force constants (eV/Å²):
atom(i)  atom(j)   xi,xj     xi,yj     xi,zj     yi,xj     yi,yj     yi,zj     zi,xj     zi,yj     zi,zj
----------------------------------------------------------------------------------------------------------
     1        1     0.0000   11.0000   22.0000   11.0000   22.0000   33.0000   22.0000   33.0000   44.0000
     1        2    33.0000   44.0000   55.0000   44.0000   55.0000   66.0000   55.0000   66.0000   77.0000
     1        3    66.0000   77.0000   88.0000   77.0000   88.0000   99.0000   88.0000   99.0000  110.0000
     1        4    99.0000  110.0000  121.0000  110.0000  121.0000  132.0000  121.0000  132.0000  143.0000
     1        5   132.0000  143.0000  154.0000  143.0000  154.0000  165.0000  154.0000  165.0000  176.0000
     1        6   165.0000  176.0000  187.0000  176.0000  187.0000  198.0000  187.0000  198.0000  209.0000
     1        7   198.0000  209.0000  220.0000  209.0000  220.0000  231.0000  220.0000  231.0000  242.0000
     2        2    66.0000   77.0000   88.0000   77.0000   88.0000   99.0000   88.0000   99.0000  110.0000
     2        3    99.0000  110.0000  121.0000  110.0000  121.0000  132.0000  121.0000  132.0000  143.0000
     2        4   132.0000  143.0000  154.0000  143.0000  154.0000  165.0000  154.0000  165.0000  176.0000
     2        5   165.0000  176.0000  187.0000  176.0000  187.0000  198.0000  187.0000  198.0000  209.0000
     2        6   198.0000  209.0000  220.0000  209.0000  220.0000  231.0000  220.0000  231.0000  242.0000
     2        7   231.0000  242.0000  253.0000  242.0000  253.0000  264.0000  253.0000  264.0000  275.0000
     3        3   132.0000  143.0000  154.0000  143.0000  154.0000  165.0000  154.0000  165.0000  176.0000
     3        4   165.0000  176.0000  187.0000  176.0000  187.0000  198.0000  187.0000  198.0000  209.0000
     3        5   198.0000  209.0000  220.0000  209.0000  220.0000  231.0000  220.0000  231.0000  242.0000
     3        6   231.0000  242.0000  253.0000  242.0000  253.0000  264.0000  253.0000  264.0000  275.0000
     3        7   264.0000  275.0000  286.0000  275.0000  286.0000  297.0000  286.0000  297.0000  308.0000
     4        4   198.0000  209.0000  220.0000  209.0000  220.0000  231.0000  220.0000  231.0000  242.0000
     4        5   231.0000  242.0000  253.0000  242.0000  253.0000  264.0000  253.0000  264.0000  275.0000
     4        6   264.0000  275.0000  286.0000  275.0000  286.0000  297.0000  286.0000  297.0000  308.0000
     4        7   297.0000  308.0000  319.0000  308.0000  319.0000  330.0000  319.0000  330.0000  341.0000
     5        5   264.0000  275.0000  286.0000  275.0000  286.0000  297.0000  286.0000  297.0000  308.0000
     5        6   297.0000  308.0000  319.0000  308.0000  319.0000  330.0000  319.0000  330.0000  341.0000
     5        7   330.0000  341.0000  352.0000  341.0000  352.0000  363.0000  352.0000  363.0000  374.0000
     6        6   330.0000  341.0000  352.0000  341.0000  352.0000  363.0000  352.0000  363.0000  374.0000
     6        7   363.0000  374.0000  385.0000  374.0000  385.0000  396.0000  385.0000  396.0000  407.0000
     7        7   396.0000  407.0000  418.0000  407.0000  418.0000  429.0000  418.0000  429.0000  440.0000
""".strip()
    assert actual == {"text/plain": reference}


def test_descriptor(Sr2TiO4, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_string": ["__str__"],
    }
    check_descriptors(Sr2TiO4, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_force_constants = raw_data.force_constant("Sr2TiO4")
    with mock_file("force_constant", raw_force_constants) as mocks:
        check_read(ForceConstant, mocks, raw_force_constants)
