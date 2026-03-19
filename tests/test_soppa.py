import pytest
from pyscf import gto, scf
from pyppm.soppa import SOPPA
import numpy as np


@pytest.fixture(scope="module")
def hf_data():
    """
    Fixture that returns the molecule and RHF wavefunction for the HF molecule.
    The SCF result is saved to a checkpoint file.
    """
    mol = gto.M(
        atom="""
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    """,
        basis="3-21g",
        unit="angstrom",
    )
    mf = scf.RHF(mol)
    mf.chkfile = "HF_test_hrpa.chk"
    mf.kernel()
    pp = SOPPA(mol=mol, chkfile=mf.chkfile, mole_name="HF_test", calc_int=True)
    return mol, mf.chkfile


@pytest.mark.parametrize("da0_sum", [1.39345969981])
def test_da0_triplet_t23_sum(da0_sum, hf_data):
    """Test for da0_triplet_t23 matrix sum

    Args:
        da0_sum (real): full da0_triplet_t23 matrix sum
    """
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    da0_ = ((pp.da0_triplet_t23) ** 2).sum()

    assert abs(da0_ - da0_sum) < 1e-5


@pytest.mark.parametrize("da0_sum", [1.486263756012])
def test_da0_singlet_s1_sum(da0_sum, hf_data):
    """Test for da0_singlet_s1 matrix sum

    Args:
        da0_sum (real): full da0_singlet_s1 matrix sum
    """
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    da0_ = ((pp.da0_singlet_s1) ** 2).sum()

    assert abs(da0_ - da0_sum) < 1e-5


@pytest.mark.parametrize("t1", [6.3372151626])
def test_t1_opt_pso_sum(t1, hf_data):
    """Test for t1_opt_pso matrix sum

    Args:
        da0_sum (real): full t1_opt_pso matrix sum
    """
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    t1_opt_pso_ = ((pp.t1_opt_PSO([0])) ** 2).sum()

    assert abs(t1_opt_pso_ - t1) < 1e-5


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"FCSD": True, "atmlst": [0]}, 14.8674117526),
        ({"FC": True, "atmlst": [0]}, 17.804784338),
    ],
)
def test_t1_opt_triplet_sq_sum(kwargs, expected, hf_data):
    """Test for squared sum of t1_opt_triplet under different configurations

    Args:
        kwargs (dict): arguments for t1_opt_triplet
        expected (real): reference sum of squares
    """
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    t1 = pp.t1_opt_triplet(**kwargs)
    t1_sq_sum = (t1**2).sum()

    assert np.isclose(t1_sq_sum, expected, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs, expected, is_scalar",
    [
        (
            {"FCSD": True, "atm1lst": [0], "atm2lst": [1]},
            -0.3087503236885935,
            False,
        ),
        (
            {"FC": True, "atm1lst": [0], "atm2lst": [1]},
            -0.4162275673613914,
            True,
        ),
        (
            {"PSO": True, "atm1lst": [0], "atm2lst": [1]},
            -0.001676545074076328,
            False,
        ),
    ],
)
def test_w4_outputs(kwargs, expected, is_scalar, hf_data):
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    w4 = pp.w4(**kwargs)

    if is_scalar:
        val = np.asarray(w4).item()
    else:
        val = w4.trace()

    assert np.isclose(val, expected, atol=1e-6)


@pytest.mark.parametrize("expected", [0.6543307347])
def test_da0_singlet_s2_sq_sum(expected, hf_data):
    """Test for squared sum (Frobenius norm^2) of da0_singlet_s2"""
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    da = pp.da0_singlet_s2
    val = (da**2).sum()

    assert np.isclose(val, expected, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"atmlst": [0], "FC": True}, 2160.8087677677577),
        ({"atmlst": [0], "FCSD": True}, 1637.3722306819564),
        ({"atmlst": [0], "PSO": True}, 48.73103992834297),
    ],
)
def test_trans_mat_1_sq_sum(kwargs, expected, hf_data):
    """Test for squared sum (Frobenius norm^2) of trans_mat_1"""
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    trans_mat = pp.trans_mat_1(**kwargs)
    val = (trans_mat**2).sum()

    assert np.isclose(val, expected, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"atmlst": [0], "FC": True}, 1795.3715255671955),
        ({"atmlst": [0], "FCSD": True}, 1348.3045237491642),
        ({"atmlst": [0], "PSO": True}, 5.909465854359097),
    ],
)
def test_trans_mat_2_sq_sum(kwargs, expected, hf_data):
    """Test for squared sum (Frobenius norm^2) of trans_mat_2"""
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    trans_mat = pp.trans_mat_2(**kwargs)
    val = (trans_mat**2).sum()

    assert np.isclose(val, expected, atol=1e-6)


@pytest.mark.parametrize(
    "triplet, expected",
    [
        (True, 30.398518814295922),
        (False, 24.23643950164012),
    ],
)
def test_communicator_sq_sum(triplet, expected, hf_data):
    """Test for squared sum (Frobenius norm^2) of Communicator"""
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    comm = pp.Communicator(triplet=triplet)
    val = (comm**2).sum()

    assert np.isclose(val, expected, atol=1e-6)


@pytest.mark.parametrize(
    "method_name, expected",
    [
        ("pp_ssc_pso", 3.170592785754413e-08),
        ("pp_ssc_fc", 2.8224521918336443e-09),
        ("pp_ssc_fcsd", -1.609854780604382e-08),
    ],
)
def test_pp_ssc_trace(method_name, expected, hf_data):
    """Test for trace of first block of pp_ssc contributions"""
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    method = getattr(pp, method_name)
    result = method(atm1lst=[0], atm2lst=[1])

    val = result[0].trace()

    assert np.isclose(val, expected, atol=1e-10)


import pytest
import numpy as np


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"atom1": "F", "atom2": "H", "PSO": True}, 151.43373764021496),
        ({"atom1": "F", "atom2": "H", "FC": True}, 13.48058592193597),
        ({"atom1": "F", "atom2": "H", "FCSD": True}, -76.88982564413479),
    ],
)
def test_ssc_values(kwargs, expected, hf_data):
    """Test for scalar SSC values under different contributions"""
    mol, chkfile = hf_data
    pp = SOPPA(mol=mol, chkfile=chkfile, mole_name="HF_test", calc_int=False)

    val = pp.ssc(**kwargs)

    val = np.asarray(val).item()

    assert np.isclose(val, expected, atol=1e-6)
