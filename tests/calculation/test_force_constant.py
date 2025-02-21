# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.force_constant import ForceConstant
from py4vasp._calculation.structure import Structure


@pytest.fixture(params=("all atoms", "selective dynamics"))
def Sr2TiO4(raw_data, request):
    raw_force_constants = raw_data.force_constant(f"Sr2TiO4 {request.param}")
    force_constants = ForceConstant.from_data(raw_force_constants)
    force_constants.ref = types.SimpleNamespace()
    structure = Structure.from_data(raw_force_constants.structure)
    force_constants.ref.structure = structure
    force_constants.ref.force_constants = raw_force_constants.force_constants
    if request.param == "all atoms":
        force_constants.ref.selective_dynamics = None
    else:
        force_constants.ref.selective_dynamics = raw_force_constants.selective_dynamics
    force_constants.ref.format_output = get_format_output(request.param)
    force_constants.ref.molden_string = get_molden_string(request.param)
    return force_constants


def test_Sr2TiO4_read(Sr2TiO4, Assert):
    actual = Sr2TiO4.read()
    reference_structure = Sr2TiO4.ref.structure.read()
    Assert.same_structure(actual["structure"], reference_structure)
    Assert.allclose(actual["force_constants"], Sr2TiO4.ref.force_constants)
    if Sr2TiO4.ref.selective_dynamics is None:
        assert "selective_dynamics" not in actual
    else:
        Assert.allclose(actual["selective_dynamics"], Sr2TiO4.ref.selective_dynamics)


def test_Sr2TiO4_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    assert actual == Sr2TiO4.ref.format_output


def get_format_output(selection):
    if selection == "all atoms":
        output = """\
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
    else:
        output = """\
Force constants (eV/Å²):
atom(i)  atom(j)   xi,xj     xi,yj     xi,zj     yi,xj     yi,yj     yi,zj     zi,xj     zi,yj     zi,zj
----------------------------------------------------------------------------------------------------------
     1        1    -7.6209  -12.7444   -6.8500  -12.7444    3.0199   -1.7746   -6.8500   -1.7746    9.8872
     1        3     frozen    frozen    7.1060    frozen    frozen   -1.6131    frozen    frozen   -6.0829
     1        4    -6.3750  -13.4549   16.0141    1.3898   11.4429    1.7273    1.0368   -3.7287   -0.1421
     1        5     1.9381    frozen    frozen   10.3378    frozen    frozen    2.0115    frozen    frozen
     1        7     frozen    1.2330  -17.3469    frozen    4.4415    3.5973    frozen  -15.3936    2.4955
     2   frozen
     3        3     frozen    frozen    frozen    frozen    frozen    frozen    frozen    frozen   -9.1394
     3        4     frozen    frozen    frozen    frozen    frozen    frozen   -5.8659    9.7352    2.2344
     3        5     frozen    frozen    frozen    frozen    frozen    frozen    2.5816    frozen    frozen
     3        7     frozen    frozen    frozen    frozen    frozen    frozen    frozen   -7.8676   11.1858
     4        4    -7.5699    1.5580   -9.0859    1.5580    4.9797   -7.5134   -9.0859   -7.5134    0.9011
     4        5    -0.5171    frozen    frozen  -11.1244    frozen    frozen    3.8899    frozen    frozen
     4        7     frozen  -12.9006   -2.2232    frozen   -6.1506   -4.5992    frozen   11.7996   -6.5990
     5        5     0.3066    frozen    frozen    frozen    frozen    frozen    frozen    frozen    frozen
     5        7     frozen   -2.7163    0.3955    frozen    frozen    frozen    frozen    frozen    frozen
     6   frozen
     7        7     frozen    frozen    frozen    frozen   13.3171   -9.1961    frozen   -9.1961    0.8190"""
    return {"text/plain": output}


def test_eigenvectors(Sr2TiO4, Assert):
    _, eigenvectors = np.linalg.eigh(Sr2TiO4.ref.force_constants.T)
    selective_dynamics = Sr2TiO4.ref.selective_dynamics
    if selective_dynamics is None:
        expected_vectors = eigenvectors.T.reshape(len(eigenvectors), -1, 3)[::-1]
    else:
        expected_vectors = np.zeros((len(eigenvectors), 7, 3))
        expected_vectors[:, selective_dynamics] = eigenvectors.T[::-1]
    actual_vectors = Sr2TiO4.eigenvectors()
    for actual, expected in zip(actual_vectors, expected_vectors):
        sign_actual = np.sign(actual.flatten()[np.argmax(np.abs(actual))])
        sign_expected = np.sign(expected.flatten()[np.argmax(np.abs(expected))])
        Assert.allclose(sign_actual * actual, sign_expected * expected, tolerance=10)


def test_to_molden(Sr2TiO4, Assert):
    molden_string = Sr2TiO4.to_molden()
    assert molden_string == Sr2TiO4.ref.molden_string


def get_molden_string(selection):
    if selection == "all atoms":
        return """\
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
Sr    14.166510     6.204469     0.000000
Sr     7.787200     3.410540     0.000000
Ti     0.000000     0.000000     0.000000
O     18.480194     8.093722     0.000000
O      3.473736     1.521383     0.000000
O      1.052770    -2.403750     2.624196
O     -1.052760     2.403754     2.624196
[FR-NORM-COORD]
vibration 1
    0.156893     0.219287    -0.196207
   -0.129069     0.199016    -0.192698
   -0.154980    -0.352561    -0.042875
   -0.218295    -0.104283     0.363516
    0.173237     0.101574    -0.134126
   -0.267478    -0.073626     0.093545
   -0.186670     0.523854    -0.035657
vibration 2
   -0.361643     0.541273     0.214260
   -0.075954    -0.111383    -0.497241
    0.114638    -0.086166    -0.048657
    0.165209     0.224463    -0.112100
   -0.023466     0.047529    -0.047162
   -0.203592     0.041963     0.142302
   -0.050444    -0.173741     0.224728
vibration 3
    0.103461    -0.052696    -0.081337
    0.126747    -0.180569    -0.071812
   -0.061367     0.029125     0.238692
   -0.044601     0.562957    -0.049518
   -0.122792     0.012704     0.018512
   -0.048074    -0.463357    -0.390507
   -0.388672     0.063611    -0.054085
vibration 4
   -0.190188    -0.085600    -0.128202
   -0.251216    -0.200344     0.296617
   -0.166911    -0.091559     0.299596
   -0.141411    -0.048178    -0.144838
    0.133532    -0.020024    -0.144217
   -0.371868    -0.109249    -0.142921
    0.310355     0.002894     0.523004
vibration 5
   -0.189094     0.214994    -0.327655
   -0.197057     0.106929     0.513600
    0.309705    -0.018210     0.005951
    0.092402     0.263582    -0.174531
    0.126717    -0.054105     0.148761
    0.067249    -0.060646     0.414495
   -0.232555     0.106225    -0.073270
vibration 6
    0.151547     0.026490     0.414118
   -0.094657    -0.105258     0.171492
    0.063678    -0.168645     0.079741
   -0.083511    -0.212795     0.019901
    0.426381     0.227128     0.410870
   -0.193818    -0.012798    -0.092874
   -0.354267    -0.294699    -0.054520
vibration 7
   -0.207331     0.045621    -0.026949
    0.172969    -0.182110    -0.030444
   -0.301482    -0.314190    -0.183294
   -0.109758    -0.174224    -0.189660
   -0.248167    -0.153324     0.585899
    0.074154    -0.304288     0.123195
    0.148070     0.154319    -0.035254
vibration 8
   -0.264536    -0.183060     0.160330
   -0.334692    -0.072420    -0.017248
   -0.123458     0.550918    -0.056159
   -0.091522     0.058924     0.142063
   -0.220458     0.037348     0.240633
   -0.386268     0.153962     0.060431
   -0.091505     0.253589    -0.205343
vibration 9
   -0.241286    -0.226187     0.447454
   -0.125377     0.019026     0.160699
   -0.415614    -0.161557    -0.161609
   -0.031397     0.279943     0.116595
    0.100056     0.146445    -0.274106
    0.367879    -0.083508     0.246783
   -0.098011     0.044334     0.096046
vibration 10
    0.129485     0.036287    -0.134760
    0.244904    -0.199476     0.087297
   -0.084613     0.086930     0.255575
    0.047215    -0.000448     0.550520
   -0.257918     0.194395     0.059491
   -0.090501    -0.116240     0.470908
    0.010594    -0.314487     0.143908
vibration 11
   -0.011557     0.123789    -0.019060
    0.344329    -0.211147     0.129983
   -0.133229    -0.066766     0.151472
   -0.361338     0.041490    -0.217152
   -0.177082     0.071462    -0.043413
    0.055314     0.640624     0.006456
   -0.304973     0.143884     0.118254
vibration 12
    0.003176    -0.187655    -0.069828
    0.240967    -0.177114     0.117942
    0.017923    -0.010397    -0.388016
    0.524927    -0.207155    -0.184980
   -0.081535     0.351179    -0.172643
   -0.268932    -0.070845    -0.003022
   -0.239581     0.186885     0.153075
vibration 13
    0.090871    -0.140628     0.369601
   -0.000424     0.252861     0.032908
    0.435632    -0.035052     0.027581
   -0.158446    -0.150623     0.029377
   -0.376908    -0.278231    -0.007986
   -0.007806    -0.171100     0.086286
   -0.223677     0.160836     0.444638
vibration 14
   -0.174089    -0.281080    -0.311027
    0.082113     0.477503    -0.275568
   -0.303365     0.124487     0.088396
    0.089277    -0.046556    -0.081965
    0.194040    -0.101557     0.208496
    0.006173     0.075534     0.067536
   -0.333445    -0.210512     0.306302
vibration 15
   -0.145516     0.207244     0.138082
    0.339989    -0.227905     0.053409
   -0.033012     0.361063     0.118603
    0.175895    -0.156317     0.214323
    0.441017    -0.381589     0.000159
    0.158225    -0.090858    -0.034879
   -0.048876     0.328250     0.121825
vibration 16
    0.221193    -0.132793     0.097692
   -0.288706    -0.183238    -0.115504
   -0.164336    -0.247021     0.454587
    0.480926    -0.080159    -0.139031
   -0.117438    -0.326597    -0.038059
   -0.004506     0.161718     0.196148
   -0.121557     0.108126    -0.169961
vibration 17
   -0.268849     0.089530     0.161737
    0.161421     0.380473     0.055937
    0.057147    -0.024642     0.518037
    0.198068    -0.088560    -0.033231
   -0.109113     0.484886     0.113744
    0.113400    -0.035910    -0.133079
    0.181643     0.256300    -0.093199
vibration 18
   -0.246708     0.007501    -0.151497
   -0.204153    -0.085434     0.127964
    0.092396    -0.262061    -0.127460
    0.256311     0.039655     0.503576
   -0.165085    -0.072239     0.201980
    0.200638     0.274589    -0.458485
   -0.094261    -0.030046     0.165958
vibration 19
    0.265219     0.223531     0.225932
    0.222392     0.386419     0.310806
   -0.281187    -0.034412    -0.147958
    0.224165     0.333480     0.037773
   -0.059633    -0.238389     0.113840
   -0.357129     0.145493    -0.082989
    0.192495    -0.031949     0.034995
vibration 20
    0.458689     0.316605    -0.035197
   -0.347832    -0.030556    -0.024568
   -0.204639     0.314277    -0.070461
    0.076392    -0.024228    -0.128262
   -0.076621     0.256751     0.203837
    0.353268    -0.023786    -0.037222
    0.011763     0.135940     0.372031
vibration 21
    0.168403    -0.381651     0.017280
    0.111705    -0.135792    -0.232138
    0.290682    -0.111689    -0.015012
    0.038561     0.413506     0.037338
    0.278135     0.129190     0.320827
   -0.010019     0.205744     0.167992
    0.296240     0.269946     0.203851
"""
    else:
        return """\
[Molden Format]
[FREQ]
  -46.354197
  -26.378556
  -15.912233
  -11.026778
   -5.992832
   -2.081234
   13.690809
   22.323949
   25.639954
   37.190779
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
    0.410207    -0.138513    -0.282935
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.061475
   -0.246503    -0.303831     0.426681
    0.048245     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.534739    -0.324507
vibration 2
   -0.289573     0.445590    -0.485282
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.043659
   -0.065861     0.503494    -0.093882
   -0.150821     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.436657     0.010896
vibration 3
   -0.208840     0.541389     0.241716
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.094700
   -0.099672    -0.302731     0.193030
    0.594447     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.161045     0.278827
vibration 4
    0.213441    -0.062541    -0.353949
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.666353
   -0.256730     0.044697     0.156886
    0.177380     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000    -0.260742     0.435034
vibration 5
   -0.269290    -0.326848    -0.152251
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.128015
   -0.103035    -0.445347    -0.308925
   -0.248138     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.330593     0.553024
vibration 6
   -0.093503    -0.032551     0.584266
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.122318
   -0.479618     0.260290     0.348155
   -0.401990     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.167889     0.158398
vibration 7
   -0.000816     0.223470    -0.151174
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.114313
    0.509866    -0.153873     0.591897
   -0.428330     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000    -0.158200     0.267699
vibration 8
   -0.212986    -0.433659     0.173663
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.306370
    0.527606     0.242020     0.188560
    0.313063     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.411555     0.051691
vibration 9
    0.362677     0.373179     0.276580
    0.000000     0.000000     0.000000
    0.000000     0.000000     0.486208
    0.266947    -0.250910    -0.361938
   -0.263377     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.280619    -0.054579
vibration 10
    0.632889    -0.009990     0.081769
    0.000000     0.000000     0.000000
    0.000000     0.000000    -0.408219
    0.093868     0.386323    -0.117105
    0.126816     0.000000     0.000000
    0.000000     0.000000     0.000000
    0.000000     0.137230     0.468335
"""


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.force_constant("Sr2TiO4 all atoms")
    check_factory_methods(ForceConstant, data)
