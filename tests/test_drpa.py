import pytest
from pyscf import gto, scf
import numpy as np
from pyppm.drpa import DRPA


@pytest.fixture(scope="module")
def hf_data():
    """
    Fixture that returns the molecule and RHF wavefunction for the HF molecule.
    The SCF result is saved to a checkpoint file.
    """
    mol = gto.M(atom="""H1 0 0 0; H2 0 2 0; O 0 1 0""", basis="cc-pvdz")
    mf = scf.DHF(mol)
    mf.chkfile = "h2o_test.chk"
    mf.kernel()
    dpa_obj = DRPA(
        mol=mol,
        chkfile=mf.chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=True,
    )
    return mol, mf.chkfile


@pytest.mark.parametrize(
    "energy_ref, occ_ref, pop1_ref, pop2_ref",
    [
        (0.249088, 0.0, 0.52489, 0.52489),
    ],
)
def test_mulliken_pop(energy_ref, occ_ref, pop1_ref, pop2_ref, hf_data):
    """
    Test for DRPA mulliken_pop output by parsing relevant numerical values.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    output = drpa_obj.mulliken_pop(60, 0.1)

    lines = output.splitlines()

    # --- Energy & Occupation ---
    header = lines[0]
    parts = header.replace(",", "").split()

    energy = float(parts[1])
    occ = float(parts[3])

    # --- Populations ---
    pop1 = float(lines[1].split(",")[-1])
    pop2 = float(lines[2].split(",")[-1])

    assert abs(energy - energy_ref) < 1e-5
    assert abs(occ - occ_ref) < 1e-8
    assert abs(pop1 - pop1_ref) < 1e-5
    assert abs(pop2 - pop2_ref) < 1e-5


@pytest.mark.parametrize(
    "corr, shield_ref",
    [
        (None, 532.0753225),
        ("TDA", 534.8825167641322),
    ],
)
def test_shield_O(corr, shield_ref, hf_data):
    """
    Test for DRPA shielding constant on Oxygen atom
    for different correlation schemes.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    if corr is None:
        shield = drpa_obj.shield(atom="O", corr=None)
    else:
        shield = drpa_obj.shield(atom="O", corr=corr)

    assert abs(shield_ref - shield) < 1e-5


@pytest.mark.parametrize(
    "corr, ssc_ref",
    [
        (None, -177.18111799),
        ("TDA", -138.37612808576523),
    ],
)
def test_ssc_O_H1(corr, ssc_ref, hf_data):
    """
    Test for DRPA spin-spin coupling (ssc) between O and H1
    for different correlation schemes.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    if corr is None:
        j = drpa_obj.ssc(atom1="O", atom2="H1", corr=None)
    else:
        j = drpa_obj.ssc(atom1="O", atom2="H1", corr=corr)

    assert abs(ssc_ref - j) < 1e-5


@pytest.mark.parametrize(
    "corr, freq, trace_ref",
    [
        ("RPA", 0.0, 6.133353599938398),
        ("RPA", 0.5, -10.88179610843369),
        ("TDA", 0.0, 4.416969089205873),
        ("TDA", 0.5, -13.890572184153404),
    ],
)
def test_pp_polarizability_lu_trace(corr, freq, trace_ref, hf_data):
    """
    Test for DRPA polarizability (LU) using the trace of the tensor
    for different correlation schemes and frequencies.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    pol = drpa_obj.pp_polarizability_lu(corr=corr, freq=freq)

    assert abs(trace_ref - pol.trace()) < 1e-5


@pytest.mark.parametrize(
    "corr, freq, trace_ref",
    [
        ("RPA", 0.0, 6.133353599938398),
        ("RPA", 0.5, -10.88179610843369),
        ("TDA", 0.0, 4.416969089205873),
        ("TDA", 0.5, -13.890572184153404),
    ],
)
def test_pp_polarizability_trace(corr, freq, trace_ref, hf_data):
    """
    Test for DRPA polarizability (LU) using the trace of the tensor
    for different correlation schemes and frequencies.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    pol = drpa_obj.pp_polarizability(corr=corr, freq=freq)

    assert abs(trace_ref - pol.trace()) < 1e-5


@pytest.mark.parametrize(
    "corr, trace_ref",
    [
        ("RPA", 2.337967200686534),
        ("TDA", 2.498695234352132),
    ],
)
def test_pp_shield_trace(corr, trace_ref, hf_data):
    """
    Test for DRPA pp_shield using the trace of the tensor
    for different correlation schemes.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    pol = drpa_obj.pp_shield(atmlst=[1], corr=corr)

    assert abs(trace_ref - pol.trace()) < 1e-5


@pytest.mark.parametrize(
    "corr, trace_ref",
    [
        ("RPA", 2.2185812200331204e-08),
        ("TDA", 1.1990382935043645e-08),
    ],
)
def test_pp_j_trace(corr, trace_ref, hf_data):
    """
    Test for DRPA pp_j using the trace of the tensor
    for different correlation schemes.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    pol = drpa_obj.pp_j(atm1lst=[0], atm2lst=[1], corr=corr)

    assert abs(trace_ref - pol.trace()) < 1e-11


@pytest.mark.parametrize(
    "corr, freq, M, ref_value",
    [
        ("RPA", 0.0, False, 709.8841362211674),
        ("TDA", 0.5, False, 4775.475426354),
        ("TDA", 0.5, True, 1354460328752.9932),
        ("RPA", 0.0, True, 1354460328323.6746),
    ],
)
def test_principal_propagator_norm(corr, freq, M, ref_value, hf_data):
    """
    Test for DRPA principal propagator using the squared Frobenius norm
    for different correlation schemes, frequencies, and M option.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    pol = drpa_obj.principal_propagator(freq=freq, corr=corr, M=M)

    value = (pol * pol.conj()).sum().real
    print(value)

    assert abs(ref_value - value) / abs(ref_value) < 1e-2


@pytest.mark.parametrize("ref_value", [193748.31231367783])
def test_pert_r_alpha_norm(ref_value, hf_data):
    """
    Test for DRPA pert_r_alpha using squared norm.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    pol = np.asarray(drpa_obj.pert_r_alpha([0])).reshape(
        1, 3, drpa_obj.nvir, drpa_obj.nocc
    )[0]

    value = (pol * pol.conj()).sum().real

    assert abs(ref_value - value) / abs(ref_value) < 1e-5


@pytest.mark.parametrize("ref_value", [358605.12007223244])
def test_pert_alpha_rg_norm(ref_value, hf_data):
    """
    Test for DRPA pert_alpha_rg using squared norm.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    pol = np.asarray(drpa_obj.pert_alpha_rg([0])).reshape(
        1, 3, drpa_obj.nvir, drpa_obj.nocc
    )[0]

    value = (pol * pol.conj()).sum().real

    assert abs(ref_value - value) / abs(ref_value) < 1e-5


@pytest.mark.parametrize("ref_value", [382975.1226082124])
def test_pert_r_norm(ref_value, hf_data):
    """
    Test for DRPA pert_r using squared norm.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    pol = np.asarray(drpa_obj.pert_r())

    value = (pol * pol.conj()).sum().real

    assert abs(ref_value - value) / abs(ref_value) < 1e-5


@pytest.mark.parametrize("ref_value", [2])
def test_obtain_atom_order(ref_value, hf_data):
    """
    Test for DRPA obtain_atom_order function.
    """
    mol, chkfile = hf_data

    drpa_obj = DRPA(
        mol=mol,
        chkfile=chkfile,
        rotations=None,
        mole_name="h2o_test",
        calc_int=False,
    )

    value = drpa_obj.obtain_atom_order(atom="O")

    assert value == ref_value
