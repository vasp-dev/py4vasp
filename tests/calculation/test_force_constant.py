# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.force_constant import ForceConstant
from py4vasp._calculation.structure import Structure


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_force_constants = raw_data.force_constant("Sr2TiO4")
    force_constants = ForceConstant.from_data(raw_force_constants)
    force_constants.ref = types.SimpleNamespace()
    structure = Structure.from_data(raw_force_constants.structure)
    force_constants.ref.structure = structure
    force_constants.ref.force_constants = raw_force_constants.force_constants
    return force_constants


def test_Sr2TiO4_read(Sr2TiO4, Assert):
    actual = Sr2TiO4.read()
    reference_structure = Sr2TiO4.ref.structure.read()
    Assert.same_structure(actual["structure"], reference_structure)
    Assert.allclose(actual["force_constants"], Sr2TiO4.ref.force_constants)


def test_Sr2TiO4_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    print(actual["text/plain"])
    reference = """\
Force constants (eV/Å²):
atom(i)  atom(j)   xi,xj     xi,yj     xi,zj     yi,xj     yi,yj     yi,zj     zi,xj     zi,yj     zi,zj
----------------------------------------------------------------------------------------------------------
     1        1    -7.6209  -12.7444   -6.8500  -12.7444    3.0199   -1.7746   -6.8500   -1.7746    9.8872
     1        2    10.3264    6.2942    4.1936    0.8850   -5.2268  -25.0729   -4.2978   -9.1255  -10.4114
     1        3     5.2982   -8.3381    7.1060   17.1818  -17.2539   -1.6131    0.7867    2.4073   -6.0829
     1        4    -6.3750  -13.4549   16.0141    1.3898   11.4429    1.7273    1.0368   -3.7287   -0.1421
     1        5     1.9381    2.1371   -6.9526   10.3378    1.7350    2.2975    2.0115    4.4279    3.9380
     1        6     2.5875   -3.0871  -13.1384  -11.3166    5.0820   15.8278    5.8210    4.8494   -3.2226
     1        7   -10.1788    1.2330  -17.3469   -2.5175    4.4415    3.5973   -5.5513  -15.3936    2.4955
     2        2    -6.9997   -7.7855   -5.5274   -7.7855  -10.3502   -6.6557   -5.5274   -6.6557   18.7039
     2        3    -2.7221    2.9424   -2.7551    8.2020   -3.2803   -8.2675    7.4347    4.4567    7.9685
     2        4     3.6003   -4.3397   -0.9032   -4.3590   -5.9844    8.1348   -5.5338   -2.4740   -8.2256
     2        5   -11.5776    0.2656    3.5798    7.4004   -1.7851   -5.0923   10.8343    1.9538    6.9996
     2        6    18.7319   -2.5890   -6.5365    8.5492    5.4252    8.2721    9.3952   -1.1485    7.8735
     2        7    -5.5002   -6.0327   -0.0473   -4.4723    9.5575   -4.2503    3.8170    1.6197    2.1938
     3        3    -3.5603   10.6040   -1.4755   10.6040    4.4127    2.7784   -1.4755    2.7784   -9.1394
     3        4     9.7955   -0.4537   -8.6573    8.7674    8.6688    1.6849   -5.8659    9.7352    2.2344
     3        5    -3.2344   -4.8884   -1.7541   -8.5449   -5.9354    1.0218    2.5816   -5.3482   -1.5802
     3        6     1.0815    2.9328    1.4060   -0.0598   11.0297   -5.4969   -7.9479   -3.4303   -9.0945
     3        7    -6.5373  -12.3860   -2.7463    4.2127   -9.7862   -7.9841   -1.8032   -7.8676   11.1858
     4        4    -7.5699    1.5580   -9.0859    1.5580    4.9797   -7.5134   -9.0859   -7.5134    0.9011
     4        5    -0.5171    0.2899   -5.1902  -11.1244    0.6860  -13.0982    3.8899    8.3701   -8.4709
     4        6     4.5753   -2.1025    7.2038    7.3419  -17.6385   -6.3054   -5.6244   -4.4384   11.7483
     4        7    -0.3541  -12.9006   -2.2232  -20.8060   -6.1506   -4.5992   -4.4605   11.7996   -6.5990
     5        5     0.3066    4.6578   -4.5894    4.6578  -17.8458   -6.0255   -4.5894   -6.0255    4.8355
     5        6    -3.6675    0.9753   -4.9685  -12.0109    2.4975    0.7492   -2.5312   -9.7144    3.1952
     5        7    -8.4787   -2.7163    0.3955   -9.6717   -6.7376   -3.7171  -10.1481  -10.9249  -15.5982
     6        6     2.4011    2.4964    5.6668    2.4964    3.9908    9.0345    5.6668    9.0345    4.2216
     6        7     4.6091  -10.1320  -15.2478    3.5288   -6.7502   -4.4871   -1.0869    0.4111    0.7146
     7        7     7.3257  -10.8380    6.9277  -10.8380   13.3171   -9.1961    6.9277   -9.1961    0.8190"""
    assert actual == {"text/plain": reference}


def test_eigenvectors(Sr2TiO4, Assert):
    _, eigenvectors = np.linalg.eigh(Sr2TiO4.ref.force_constants.T)
    expected_vectors = eigenvectors.T.reshape(len(eigenvectors), -1, 3)[::-1]
    Assert.allclose(Sr2TiO4.eigenvectors(), expected_vectors)


def test_to_molden(Sr2TiO4, Assert):
    molden_string = Sr2TiO4.to_molden()
    print(molden_string)
    assert molden_string == """\
[Molden Format]
[FREQ]
  -62.791407
  -54.777660
  -42.783193
  -40.543809
  -29.497268
  -27.556791
  -22.441675
  -14.356255
  -11.526163
   -5.600826
    2.601341
    4.479327
    5.905017
   10.861266
   12.843380
   20.371554
   37.020320
   41.501937
   46.211068
   51.327174
   62.717108
[FR-COORD]

[FR-NORM-COORD]

"""

def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.force_constant("Sr2TiO4")
    check_factory_methods(ForceConstant, data)
