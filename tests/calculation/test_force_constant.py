# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.force_constant import ForceConstant, ForceConstantHandler
from py4vasp._calculation.phonon_mode import PhononMode, PhononModeHandler
from py4vasp._calculation.structure import StructureHandler
from py4vasp._demo import showcase
from py4vasp._util import convert, masses


@pytest.fixture(params=("all atoms", "selective dynamics"))
def Sr2TiO4(raw_data, request):
    raw_force_constants = raw_data.force_constant(f"Sr2TiO4 {request.param}")
    force_constants = ForceConstantHandler.from_data(raw_force_constants)
    force_constants.ref = types.SimpleNamespace()
    structure = StructureHandler.from_data(raw_force_constants.structure)
    force_constants.ref.structure = structure
    # VASP stores the derivative of the force, py4vasp reports the Hessian
    force_constants.ref.force_constants = -np.array(raw_force_constants.force_constants)
    if request.param == "all atoms":
        force_constants.ref.selective_dynamics = None
    else:
        force_constants.ref.selective_dynamics = raw_force_constants.selective_dynamics
    force_constants.ref.format_output = get_format_output(request.param)
    force_constants.ref.molden_string = get_molden_string(request.param)
    return force_constants


def test_Sr2TiO4_read(Sr2TiO4, Assert):
    actual = Sr2TiO4.to_dict()
    reference_structure = Sr2TiO4.ref.structure.to_dict()
    Assert.same_structure(actual["structure"], reference_structure)
    Assert.allclose(actual["force_constants"], Sr2TiO4.ref.force_constants)
    if Sr2TiO4.ref.selective_dynamics is None:
        assert "selective_dynamics" not in actual
    else:
        Assert.allclose(actual["selective_dynamics"], Sr2TiO4.ref.selective_dynamics)


@pytest.fixture(params=("all atoms", "selective dynamics"))
def dispatcher(raw_data, request):
    raw_force_constants = raw_data.force_constant(f"Sr2TiO4 {request.param}")
    force_constants = ForceConstant.from_data(raw_force_constants)
    force_constants.ref = types.SimpleNamespace()
    force_constants.ref.format_output = get_format_output(request.param)
    force_constants.ref.selection = f"Sr2TiO4 {request.param}"
    return force_constants


def test_Sr2TiO4_print(Sr2TiO4):
    actual = str(Sr2TiO4)
    assert actual == Sr2TiO4.ref.format_output["text/plain"]


def test_print_dispatcher(dispatcher, format_):
    actual, _ = format_(dispatcher)
    assert actual == {"text/plain": dispatcher.ref.format_output["text/plain"]}


def get_format_output(selection):
    if selection == "all atoms":
        output = """\
Force constants (eV/Å²):
atom(i)  atom(j)   xi,xj     xi,yj     xi,zj     yi,xj     yi,yj     yi,zj     zi,xj     zi,yj     zi,zj
----------------------------------------------------------------------------------------------------------
     1        1     7.6209   12.7444    6.8500   12.7444   -3.0199    1.7746    6.8500    1.7746   -9.8872
     1        2   -10.3264   -6.2942   -4.1936   -0.8850    5.2268   25.0729    4.2978    9.1255   10.4114
     1        3    -5.2982    8.3381   -7.1060  -17.1818   17.2539    1.6131   -0.7867   -2.4073    6.0829
     1        4     6.3750   13.4549  -16.0141   -1.3898  -11.4429   -1.7273   -1.0368    3.7287    0.1421
     1        5    -1.9381   -2.1371    6.9526  -10.3378   -1.7350   -2.2975   -2.0115   -4.4279   -3.9380
     1        6    -2.5875    3.0871   13.1384   11.3166   -5.0820  -15.8278   -5.8210   -4.8494    3.2226
     1        7    10.1788   -1.2330   17.3469    2.5175   -4.4415   -3.5973    5.5513   15.3936   -2.4955
     2        2     6.9997    7.7855    5.5274    7.7855   10.3502    6.6557    5.5274    6.6557  -18.7039
     2        3     2.7221   -2.9424    2.7551   -8.2020    3.2803    8.2675   -7.4347   -4.4567   -7.9685
     2        4    -3.6003    4.3397    0.9032    4.3590    5.9844   -8.1348    5.5338    2.4740    8.2256
     2        5    11.5776   -0.2656   -3.5798   -7.4004    1.7851    5.0923  -10.8343   -1.9538   -6.9996
     2        6   -18.7319    2.5890    6.5365   -8.5492   -5.4252   -8.2721   -9.3952    1.1485   -7.8735
     2        7     5.5002    6.0327    0.0473    4.4723   -9.5575    4.2503   -3.8170   -1.6197   -2.1938
     3        3     3.5603  -10.6040    1.4755  -10.6040   -4.4127   -2.7784    1.4755   -2.7784    9.1394
     3        4    -9.7955    0.4537    8.6573   -8.7674   -8.6688   -1.6849    5.8659   -9.7352   -2.2344
     3        5     3.2344    4.8884    1.7541    8.5449    5.9354   -1.0218   -2.5816    5.3482    1.5802
     3        6    -1.0815   -2.9328   -1.4060    0.0598  -11.0297    5.4969    7.9479    3.4303    9.0945
     3        7     6.5373   12.3860    2.7463   -4.2127    9.7862    7.9841    1.8032    7.8676  -11.1858
     4        4     7.5699   -1.5580    9.0859   -1.5580   -4.9797    7.5134    9.0859    7.5134   -0.9011
     4        5     0.5171   -0.2899    5.1902   11.1244   -0.6860   13.0982   -3.8899   -8.3701    8.4709
     4        6    -4.5753    2.1025   -7.2038   -7.3419   17.6385    6.3054    5.6244    4.4384  -11.7483
     4        7     0.3541   12.9006    2.2232   20.8060    6.1506    4.5992    4.4605  -11.7996    6.5990
     5        5    -0.3066   -4.6578    4.5894   -4.6578   17.8458    6.0255    4.5894    6.0255   -4.8355
     5        6     3.6675   -0.9753    4.9685   12.0109   -2.4975   -0.7492    2.5312    9.7144   -3.1952
     5        7     8.4787    2.7163   -0.3955    9.6717    6.7376    3.7171   10.1481   10.9249   15.5982
     6        6    -2.4011   -2.4964   -5.6668   -2.4964   -3.9908   -9.0345   -5.6668   -9.0345   -4.2216
     6        7    -4.6091   10.1320   15.2478   -3.5288    6.7502    4.4871    1.0869   -0.4111   -0.7146
     7        7    -7.3257   10.8380   -6.9277   10.8380  -13.3171    9.1961   -6.9277    9.1961   -0.8190"""
    else:
        output = """\
Force constants (eV/Å²):
atom(i)  atom(j)   xi,xj     xi,yj     xi,zj     yi,xj     yi,yj     yi,zj     zi,xj     zi,yj     zi,zj
----------------------------------------------------------------------------------------------------------
     1        1     7.6209   12.7444    6.8500   12.7444   -3.0199    1.7746    6.8500    1.7746   -9.8872
     1        3     frozen    frozen   -7.1060    frozen    frozen    1.6131    frozen    frozen    6.0829
     1        4     6.3750   13.4549  -16.0141   -1.3898  -11.4429   -1.7273   -1.0368    3.7287    0.1421
     1        5    -1.9381    frozen    frozen  -10.3378    frozen    frozen   -2.0115    frozen    frozen
     1        7     frozen   -1.2330   17.3469    frozen   -4.4415   -3.5973    frozen   15.3936   -2.4955
     2   frozen
     3        3     frozen    frozen    frozen    frozen    frozen    frozen    frozen    frozen    9.1394
     3        4     frozen    frozen    frozen    frozen    frozen    frozen    5.8659   -9.7352   -2.2344
     3        5     frozen    frozen    frozen    frozen    frozen    frozen   -2.5816    frozen    frozen
     3        7     frozen    frozen    frozen    frozen    frozen    frozen    frozen    7.8676  -11.1858
     4        4     7.5699   -1.5580    9.0859   -1.5580   -4.9797    7.5134    9.0859    7.5134   -0.9011
     4        5     0.5171    frozen    frozen   11.1244    frozen    frozen   -3.8899    frozen    frozen
     4        7     frozen   12.9006    2.2232    frozen    6.1506    4.5992    frozen  -11.7996    6.5990
     5        5    -0.3066    frozen    frozen    frozen    frozen    frozen    frozen    frozen    frozen
     5        7     frozen    2.7163   -0.3955    frozen    frozen    frozen    frozen    frozen    frozen
     6   frozen
     7        7     frozen    frozen    frozen    frozen  -13.3171    9.1961    frozen    9.1961   -0.8190"""
    return {"text/plain": output}


def test_eigenvectors(Sr2TiO4, Assert):
    _, eigenvectors = np.linalg.eigh(Sr2TiO4.ref.force_constants.T)
    selective_dynamics = Sr2TiO4.ref.selective_dynamics
    if selective_dynamics is None:
        expected_vectors = eigenvectors.T.reshape(len(eigenvectors), -1, 3)
    else:
        expected_vectors = np.zeros((len(eigenvectors), 7, 3))
        expected_vectors[:, selective_dynamics] = eigenvectors.T
    actual_vectors = Sr2TiO4.eigenvectors()
    for actual, expected in zip(actual_vectors, expected_vectors):
        sign_actual = np.sign(actual.flatten()[np.argmax(np.abs(actual))])
        sign_expected = np.sign(expected.flatten()[np.argmax(np.abs(expected))])
        Assert.allclose(sign_actual * actual, sign_expected * expected)


def test_force_constants_are_symmetrized(raw_data, Assert):
    # the Hessian is symmetric by construction, VASP may deviate from it by numerical
    # noise, and read, print and eigenvectors must not each make their own choice
    raw_force_constant = raw_data.force_constant("Sr2TiO4 all atoms")
    asymmetric = np.array(raw_force_constant.force_constants)
    asymmetric[0, 1] += 1.0
    raw_force_constant = dataclasses.replace(
        raw_force_constant, force_constants=asymmetric
    )
    actual = ForceConstantHandler.from_data(raw_force_constant).to_dict()
    Assert.allclose(actual["force_constants"], -0.5 * (asymmetric + asymmetric.T))


def test_eigenvectors_diagonalize_reported_force_constants(Sr2TiO4, Assert):
    # eigenvectors and read must agree on the sign, so the eigenvectors come out in
    # ascending order of the eigenvalues of the matrix that read reports
    force_constants = Sr2TiO4.to_dict()["force_constants"]
    eigenvectors = Sr2TiO4.eigenvectors()
    if Sr2TiO4.ref.selective_dynamics is not None:
        eigenvectors = eigenvectors[:, Sr2TiO4.ref.selective_dynamics]
    eigenvectors = eigenvectors.reshape(len(eigenvectors), -1)
    eigenvalues = [vector @ force_constants @ vector for vector in eigenvectors]
    assert np.all(np.diff(eigenvalues) > 0)


def test_eigenvalues(Sr2TiO4, Assert):
    # VASP stores only the degrees of freedom selective dynamics leaves free, so there
    # is one eigenvalue per free direction of an atom
    expected = np.linalg.eigvalsh(Sr2TiO4.ref.force_constants)
    Assert.allclose(Sr2TiO4.eigenvalues(), expected)


def test_eigenvalues_belong_to_the_eigenvectors(Sr2TiO4, Assert):
    force_constants = Sr2TiO4.to_dict()["force_constants"]
    eigenvectors = Sr2TiO4.eigenvectors()
    if Sr2TiO4.ref.selective_dynamics is not None:
        eigenvectors = eigenvectors[:, Sr2TiO4.ref.selective_dynamics]
    eigenvectors = eigenvectors.reshape(len(eigenvectors), -1)
    expected = [vector @ force_constants @ vector for vector in eigenvectors]
    Assert.allclose(Sr2TiO4.eigenvalues(), expected)


def test_eigenvalues_dispatcher(dispatcher, raw_data, Assert):
    raw_force_constant = raw_data.force_constant(dispatcher.ref.selection)
    handler = ForceConstantHandler.from_data(raw_force_constant)
    Assert.allclose(dispatcher.eigenvalues(), handler.eigenvalues())


def expected_frequencies(Sr2TiO4, masses_per_atom):
    # the dynamical matrix divides the force constants by the square root of the
    # masses; its eigenvalues are (ħω)²/ħ², and a negative one is an unstable mode
    masses_per_direction = np.repeat(masses_per_atom, 3)
    if Sr2TiO4.ref.selective_dynamics is not None:
        selective_dynamics = np.array(Sr2TiO4.ref.selective_dynamics, dtype=np.bool_)
        masses_per_direction = masses_per_direction[selective_dynamics.flatten()]
    inverse_sqrt_mass = 1 / np.sqrt(masses_per_direction)
    weights = np.outer(inverse_sqrt_mass, inverse_sqrt_mass)
    eigenvalues = np.linalg.eigvalsh(weights * Sr2TiO4.ref.force_constants)
    squared = convert.HBAR_SQUARED * eigenvalues
    return np.where(squared < 0, 1j * np.sqrt(np.abs(squared)), np.sqrt(np.abs(squared)))


def test_frequencies(Sr2TiO4, Assert):
    elements = Sr2TiO4.ref.structure._stoichiometry().elements()
    expected = expected_frequencies(Sr2TiO4, masses.of(elements))
    Assert.allclose(Sr2TiO4.frequencies(), expected)


def test_frequencies_are_imaginary_for_unstable_modes(Sr2TiO4):
    # the random test data are not at an energy minimum, so they contain unstable
    # modes; VASP reports those as imaginary and py4vasp does not clamp them
    frequencies = Sr2TiO4.frequencies()
    unstable = Sr2TiO4.eigenvalues() < 0
    assert np.any(unstable)
    assert np.all(frequencies[unstable].real == 0)
    assert np.all(frequencies[unstable].imag > 0)
    assert np.all(frequencies[~unstable].imag == 0)


def test_frequencies_accept_custom_masses(Sr2TiO4, Assert):
    custom_masses = np.linspace(1.0, 7.0, 7)
    expected = expected_frequencies(Sr2TiO4, custom_masses)
    Assert.allclose(Sr2TiO4.frequencies(masses=custom_masses), expected)


def test_frequencies_raise_error_if_masses_do_not_match_the_atoms(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.frequencies(masses=[1.0, 2.0])


def test_frequencies_agree_with_the_phonon_modes(Assert):
    # the showcase force constants are built from the showcase modes, so both routes to
    # the frequency have to agree, including the vanishing acoustic modes
    force_constant = ForceConstant.from_data(showcase.phonon.force_constant_Sr2TiO4())
    phonon_mode = PhononMode.from_data(showcase.phonon.mode_Sr2TiO4())
    np.testing.assert_allclose(
        force_constant.frequencies(), phonon_mode.frequencies(), atol=1e-6
    )


def test_frequencies_dispatcher(dispatcher, raw_data, Assert):
    raw_force_constant = raw_data.force_constant(dispatcher.ref.selection)
    handler = ForceConstantHandler.from_data(raw_force_constant)
    masses_ = np.arange(1.0, 8.0)
    Assert.allclose(dispatcher.frequencies(masses_), handler.frequencies(masses_))


def default_masses(Sr2TiO4):
    return masses.of(Sr2TiO4.ref.structure._stoichiometry().elements())


def test_displacements_are_mass_normalized(Sr2TiO4, Assert):
    # the same normalization the phonon modes use: the normal coordinate is 1
    displacements = Sr2TiO4.displacements()
    weighted = default_masses(Sr2TiO4)[:, np.newaxis] * displacements**2
    Assert.allclose(np.sum(weighted, axis=(1, 2)), np.ones(len(displacements)))


def test_displacements_solve_the_equation_of_motion(Sr2TiO4, Assert):
    # a normal mode u with frequency ω satisfies Φ u = ω² M u, and the order is the
    # one of the frequencies
    displacements = Sr2TiO4.displacements()
    omega_squared = (Sr2TiO4.frequencies() ** 2).real / convert.HBAR_SQUARED
    masses_per_direction = np.repeat(default_masses(Sr2TiO4), 3)
    free = np.ones(displacements.shape[1:], dtype=np.bool_)
    if Sr2TiO4.ref.selective_dynamics is not None:
        free = np.array(Sr2TiO4.ref.selective_dynamics, dtype=np.bool_)
    masses_per_direction = masses_per_direction[free.flatten()]
    for displacement, eigenvalue in zip(displacements, omega_squared):
        vector = displacement[free]
        force = Sr2TiO4.ref.force_constants @ vector
        Assert.allclose(force, eigenvalue * masses_per_direction * vector)


def test_frozen_atoms_do_not_move(Sr2TiO4, Assert):
    if Sr2TiO4.ref.selective_dynamics is None:
        pytest.skip("every atom is displaced")
    frozen = ~np.array(Sr2TiO4.ref.selective_dynamics, dtype=np.bool_)
    displacements = Sr2TiO4.displacements()
    assert np.all(displacements[:, frozen] == 0)


def test_displacements_accept_custom_masses(Sr2TiO4, Assert):
    custom_masses = np.linspace(1.0, 7.0, 7)
    displacements = Sr2TiO4.displacements(masses=custom_masses)
    weighted = custom_masses[:, np.newaxis] * displacements**2
    Assert.allclose(np.sum(weighted, axis=(1, 2)), np.ones(len(displacements)))


def test_displacements_agree_with_the_phonon_modes(Assert):
    # the three translations are degenerate, so only the optical modes are unique up
    # to their sign
    force_constant = ForceConstant.from_data(showcase.phonon.force_constant_Sr2TiO4())
    phonon_mode = PhononModeHandler.from_data(showcase.phonon.mode_Sr2TiO4())
    actual = force_constant.displacements()[3:]
    expected = phonon_mode.displacements()[3:]
    for mode_actual, mode_expected in zip(actual, expected):
        sign = np.sign(np.sum(mode_actual * mode_expected))
        np.testing.assert_allclose(sign * mode_actual, mode_expected, atol=1e-8)


def test_displacements_dispatcher(dispatcher, raw_data, Assert):
    raw_force_constant = raw_data.force_constant(dispatcher.ref.selection)
    handler = ForceConstantHandler.from_data(raw_force_constant)
    masses_ = np.arange(1.0, 8.0)
    Assert.allclose(dispatcher.displacements(masses_), handler.displacements(masses_))


def test_to_molden(Sr2TiO4, Assert):
    molden_string = Sr2TiO4.to_molden()
    assert molden_string == Sr2TiO4.ref.molden_string


def parse_molden(molden_string):
    sections = {}
    for block in molden_string.split("[")[1:]:
        name, content = block.split("]", maxsplit=1)
        sections[name] = content.strip().splitlines()
    frequencies = np.array([float(line) for line in sections["FREQ"]])
    vibrations = []
    for line in sections["FR-NORM-COORD"]:
        if line.startswith("vibration"):
            vibrations.append([])
        else:
            vibrations[-1].append([float(x) for x in line.split()])
    return frequencies, np.array(vibrations)


def signed_wavenumbers(frequencies):
    # molden marks an imaginary frequency with a negative number
    signed = np.where(frequencies.imag > 0, -frequencies.imag, frequencies.real)
    return signed * convert.EV_TO_CM1


def test_to_molden_frequency_in_cm1(Sr2TiO4):
    frequencies, _ = parse_molden(Sr2TiO4.to_molden())
    expected = signed_wavenumbers(Sr2TiO4.frequencies())
    np.testing.assert_allclose(frequencies, expected, atol=1e-6)


def test_to_molden_reports_the_frequencies_of_a_stable_structure():
    # the example from the backlog in reverse: a viewer reads the label off [FREQ], so
    # it has to be the wavenumber in cm⁻¹ and not the eigenvalue of the force constants
    force_constant = ForceConstant.from_data(showcase.phonon.force_constant_Sr2TiO4())
    phonon_mode = PhononModeHandler.from_data(showcase.phonon.mode_Sr2TiO4())
    frequencies, _ = parse_molden(force_constant.to_molden())
    expected = phonon_mode.frequencies().real * convert.EV_TO_CM1
    np.testing.assert_allclose(frequencies, expected, atol=1e-3)


def test_to_molden_vectors_are_normal_modes(Sr2TiO4, Assert):
    _, vibrations = parse_molden(Sr2TiO4.to_molden())
    displacements = Sr2TiO4.displacements()
    for vibration, displacement in zip(vibrations, displacements):
        # molden only needs the direction, so the vector is normalized to 1
        Assert.allclose(np.linalg.norm(vibration), 1, tolerance=1e8)
        direction = displacement / np.linalg.norm(displacement)
        sign = np.sign(np.sum(vibration * direction))
        np.testing.assert_allclose(vibration, sign * direction, atol=1e-6)


def test_to_molden_custom_masses(Sr2TiO4):
    custom_masses = np.linspace(1.0, 7.0, 7)
    frequencies, _ = parse_molden(Sr2TiO4.to_molden(masses=custom_masses))
    expected = signed_wavenumbers(Sr2TiO4.frequencies(custom_masses))
    np.testing.assert_allclose(frequencies, expected, atol=1e-6)


def test_to_molden_dispatcher(dispatcher, raw_data):
    raw_force_constant = raw_data.force_constant(dispatcher.ref.selection)
    handler = ForceConstantHandler.from_data(raw_force_constant)
    masses_ = np.arange(1.0, 8.0)
    assert dispatcher.to_molden(masses_) == handler.to_molden(masses_)


def get_molden_string(selection):
    if selection == "all atoms":
        return """\
[Molden Format]
[FREQ]
 -888.553757
 -818.043705
 -723.484416
 -595.004432
 -587.953345
 -489.153734
 -383.292874
 -336.293581
 -276.550860
 -213.967153
  143.854920
  203.816908
  211.091063
  237.124061
  323.496001
  432.947508
  492.329141
  555.251184
  739.592354
  779.112346
  932.477636
[FR-COORD]
Sr    14.166510     6.204469     0.000000
Sr     7.787200     3.410540     0.000000
Ti     0.000000     0.000000     0.000000
O     18.480194     8.093722     0.000000
O      3.473736     1.521383     0.000000
O      1.052770    -2.403750     2.624196
O     -1.052760     2.403754     2.624196
[FR-NORM-COORD]
vibration 1
    0.059569     0.031285    -0.046619
   -0.042076     0.044961    -0.021168
   -0.099253    -0.117513    -0.003273
   -0.282405    -0.071695     0.440176
    0.168255     0.131922    -0.110473
   -0.327138    -0.133821     0.020232
   -0.298536     0.640121    -0.067890
vibration 2
    0.003661     0.001563    -0.003114
    0.029472    -0.034214    -0.017017
    0.027895     0.030357     0.058162
    0.050403     0.658423    -0.129848
   -0.121289     0.009743     0.098977
    0.091960    -0.429079    -0.245193
   -0.483178     0.001567    -0.169968
vibration 3
   -0.057758     0.016438    -0.010913
   -0.050939    -0.068309    -0.026856
   -0.038918    -0.043650     0.160968
   -0.046648     0.142389    -0.143661
    0.076866     0.033899    -0.390764
   -0.410829    -0.168486    -0.333080
    0.203117    -0.089229     0.631499
vibration 4
    0.031907    -0.024326     0.058623
   -0.011590    -0.065995     0.061237
   -0.017099    -0.077266     0.061623
   -0.154595    -0.343464    -0.157009
    0.264477     0.055686     0.695167
   -0.315099    -0.163549    -0.253036
   -0.133004    -0.214104    -0.015358
vibration 5
    0.057046    -0.189202    -0.092753
    0.064051    -0.019280     0.051717
   -0.150768     0.028681     0.013935
   -0.259361    -0.113804    -0.101703
   -0.339564    -0.334543     0.063214
    0.216739    -0.256046    -0.392239
    0.447503     0.349648    -0.103765
vibration 6
   -0.229730     0.161073    -0.080264
   -0.025395    -0.051208    -0.040472
   -0.036138    -0.142059    -0.091241
    0.026055     0.026680    -0.158434
   -0.433390    -0.214602     0.402126
   -0.146032    -0.287404     0.498974
    0.124926     0.228263     0.193469
vibration 7
   -0.022610    -0.019393    -0.163215
   -0.081377     0.109923     0.471754
    0.071529    -0.275744     0.070548
    0.023847     0.109498    -0.227039
    0.500075    -0.117521    -0.144906
    0.396210    -0.190042     0.262133
   -0.063696     0.037723     0.169725
vibration 8
   -0.093105    -0.056989    -0.165011
   -0.210848    -0.018668     0.184161
    0.222685     0.539974     0.116859
   -0.045132     0.054602    -0.085883
   -0.195180    -0.052517     0.127571
   -0.490593     0.353986     0.008158
   -0.114171     0.248578    -0.130066
vibration 9
   -0.027043    -0.167928     0.113555
    0.016930    -0.118030     0.143128
   -0.269402     0.121283     0.148680
   -0.038499     0.119211     0.574392
   -0.209769     0.232217     0.050823
   -0.001604    -0.209803     0.465995
    0.016139    -0.294078     0.141540
vibration 10
    0.312784     0.146556    -0.376672
    0.244986    -0.028690    -0.034806
    0.292027     0.005877     0.275683
    0.144983    -0.149486     0.389178
   -0.213598    -0.001678     0.094123
   -0.150207    -0.261324     0.202871
    0.123927    -0.330271     0.098389
vibration 11
   -0.053094     0.132830    -0.017641
    0.364727    -0.304888     0.167565
   -0.146152    -0.011377     0.005394
   -0.205510     0.009565    -0.272504
   -0.056544     0.140124    -0.124203
    0.001218     0.626736    -0.080522
   -0.287935     0.255525     0.036348
vibration 12
   -0.018386    -0.208144    -0.153255
    0.067709    -0.106948     0.018374
    0.008475     0.015027    -0.452385
    0.606050    -0.156399    -0.072974
    0.103803     0.348903    -0.132763
   -0.280395    -0.276556    -0.023365
   -0.082649     0.021045     0.037410
vibration 13
    0.104754    -0.077336     0.341504
    0.123580     0.047495     0.135492
    0.473099    -0.035153    -0.087863
   -0.028850    -0.172820     0.002623
   -0.450459    -0.143838    -0.094954
   -0.139797    -0.186482     0.080521
   -0.156772     0.306543     0.393807
vibration 14
    0.198679     0.235846     0.118512
   -0.286007    -0.435310     0.106965
    0.113290    -0.059113    -0.125230
    0.121040     0.078143     0.144438
    0.015673    -0.003231    -0.180312
    0.089975    -0.222974    -0.077928
    0.476835     0.089676    -0.449915
vibration 15
   -0.165751     0.191262     0.081355
    0.185163     0.060776     0.055153
    0.010691     0.348844    -0.053471
    0.123358    -0.102106     0.307609
    0.530462    -0.182278     0.009859
    0.099780    -0.458288    -0.178290
    0.248849     0.121005    -0.017961
vibration 16
   -0.055254    -0.149427     0.021167
   -0.011478    -0.180962    -0.142135
    0.065111    -0.023754     0.321559
    0.473761    -0.250838    -0.043415
    0.319781    -0.427893    -0.040235
    0.187387    -0.006498     0.204407
   -0.228724     0.327731     0.000804
vibration 17
    0.305847     0.133138     0.076945
   -0.032225     0.113909     0.054521
   -0.394900     0.193856    -0.011758
    0.502533    -0.029472    -0.416954
   -0.064423    -0.251568     0.057501
   -0.152342    -0.018785     0.189142
   -0.139244     0.174871     0.254667
vibration 18
   -0.086871     0.030075     0.070216
    0.053965     0.113018     0.081617
   -0.026428    -0.151731     0.254196
    0.501426    -0.051303    -0.066249
   -0.241640     0.383679    -0.059773
   -0.262163     0.000229    -0.156713
    0.264400     0.142447    -0.474160
vibration 19
    0.061728    -0.072094     0.038222
    0.080122    -0.004823    -0.068464
    0.049477     0.068805     0.041137
   -0.330054     0.194986    -0.461192
    0.309509     0.194018    -0.089110
   -0.212168    -0.181609     0.490525
    0.295520     0.090599    -0.233114
vibration 20
   -0.004909     0.053793    -0.022328
   -0.068050    -0.012752    -0.040223
    0.017239     0.125336     0.087199
   -0.083255    -0.352964    -0.218246
   -0.128128     0.632893     0.000447
    0.480826    -0.220223    -0.020326
   -0.113896     0.177553     0.214110
vibration 21
    0.052519    -0.038772     0.006672
    0.010701    -0.000267    -0.022253
    0.062077    -0.019385    -0.006276
    0.195184     0.434036     0.161397
    0.184969     0.239149     0.470359
    0.136103     0.283874    -0.060584
    0.328024     0.329683     0.326449
"""
    else:
        return """\
[Molden Format]
[FREQ]
 -781.290252
 -531.046836
 -354.656674
 -266.989799
 -238.462771
 -113.899665
  375.639761
  452.456300
  512.717043
  607.758215
[FR-COORD]
Sr    14.166510     6.204469     0.000000
Sr     7.787200     3.410540     0.000000
Ti     0.000000     0.000000     0.000000
O     18.480194     8.093722     0.000000
O      3.473736     1.521383     0.000000
O      1.052770    -2.403750     2.624196
O     -1.052760     2.403754     2.624196
[FR-NORM-COORD]
vibration 1
    0.097404    -0.006529    -0.056010
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.063693
   -0.308667    -0.319899     0.482548
    0.097644     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.681848    -0.279932
vibration 2
   -0.060376     0.035870    -0.112866
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.012721
   -0.034455     0.675778    -0.119338
   -0.552686     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.335116    -0.303807
vibration 3
   -0.211870     0.238256    -0.118368
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.259150
   -0.384202     0.068460     0.029493
    0.211502     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.302683     0.726400
vibration 4
    0.294464    -0.226841    -0.098960
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.572227
   -0.372173     0.262800     0.367590
   -0.208130     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000    -0.342621     0.145571
vibration 5
    0.136318     0.358413     0.055349
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.097219
    0.128301     0.403199     0.303429
    0.499596     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000    -0.277167    -0.492874
vibration 6
   -0.102327     0.001845     0.634808
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.052864
   -0.453270     0.248152     0.369119
   -0.384217     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.135118     0.120777
vibration 7
   -0.321012    -0.267731     0.001016
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.177765
    0.017443     0.283070     0.118581
    0.691801     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.053383    -0.466623
vibration 8
   -0.150152     0.005995    -0.103483
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.151742
    0.152695    -0.009016     0.817911
   -0.283262     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000    -0.397189     0.115383
vibration 9
   -0.059012     0.056861     0.051560
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.346067
    0.750104    -0.356326     0.075322
   -0.287509     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.278128    -0.124693
vibration 10
    0.124365    -0.091703     0.037045
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.115232
    0.500074     0.462575     0.200124
    0.286184     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.349719     0.503161
"""


def test_print_writes_to_stdout(dispatcher, capsys):
    assert dispatcher.print() is None
    assert capsys.readouterr().out == str(dispatcher) + "\n"


def test_selections(dispatcher):
    assert dispatcher.selections() == {"force_constant": ["default"]}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.force_constant("Sr2TiO4 all atoms")
    check_factory_methods(ForceConstant, data, skip_methods=["selections"])
