import pytest
from pyscf import gto, scf
from PyPPM.ssc_pol_prop import Prop_pol


@pytest.mark.parametrize(" Triplet, M_trace ", [(True, [534.94666413])])
def test_M(Triplet, M_trace):
    """
    Test for Inverse of the Principal Propagator Matrix
    it use the trace

    Args:
        Triplet (boolean): if the response is triplet or singlet
        M_trace (real) : value of the M trace
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    m = Prop_pol(mf).M(triplet=Triplet)
    assert M_trace - m.trace() < 1e-5


@pytest.mark.parametrize("Element, I", [("F1", 1)])
def test_obtain_atom_order(Element, I):
    """Test for the function that gives the id of the element chosen

    Args:
        Element (str): Element
        ID (int): id number of the atom in the molecule
    """
    HF_mol = gto.M(atom="""H 0 0 0; F1 1 0 0""", basis="sto-3g")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    prop = Prop_pol(mf)
    id = prop.obtain_atom_order(Element)
    assert id == I


@pytest.mark.parametrize(" atm_id, pert_fc_sum ", [([0], [-0.26329630])])
def test_pert_fc(atm_id, pert_fc_sum):
    """
    Test for Perturbator
    it use the trace

    Args:
        atm_id (list): list with the id of the atom in wich the perturbator is centered
        pert_fc_sum (real) : perturbator sum
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    pert = Prop_pol(mf).pert_fc(atm_id)
    assert pert_fc_sum - pert[0].sum() < 1e-5


@pytest.mark.parametrize(" atm_id, pert_pso_squared_sum ", [([1], [38.983022])])
def test_pert_pso(atm_id, pert_pso_squared_sum):
    """Test for the PSO perturbator
    It uses the sum of squared of the perturbator in one direction because
    the perturbator sum gives differents results in consecutive calculations
    Args:
        atm_id (list): ist with the id of the atom in wich the perturbator is centered
        pert_pso_squared_sum (real): sum of squared of the pso perturbator
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    pert_pso = Prop_pol(mf).pert_pso(atm_id)
    pert_pso_squared_sum_ = (pert_pso[0][2] * pert_pso[0][2]).sum()
    assert pert_pso_squared_sum - pert_pso_squared_sum_ < 1e-3

@pytest.mark.parametrize(" atm_id, fcsd_integrals ", [(1, [1842.4910058456317])])
def test_get_integrals_fcsd(atm_id, fcsd_integrals):
    """Test for the fcsd integrals
    Args:
        atm_id (list): ist with the id of the atom in wich the perturbator is centered
        pert_fcsd_squared_sum (real): sum of squared of the fcsd perturbator
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    fcsd_integrals_ = Prop_pol(mf).get_integrals_fcsd(atm_id)
    assert fcsd_integrals - fcsd_integrals_.sum() < 1e-2

@pytest.mark.parametrize(" atm_id, pert_fcsd_squared_sum ", [([0], [1.5196725923351735])])
def test_pert_fcsd(atm_id, pert_fcsd_squared_sum):
    """Test for the fcsd perturbator
    It uses the sum of squared of the perturbator in one direction because
    the perturbator sum gives differents results in consecutive calculations
    Args:
        atm_id (list): ist with the id of the atom in wich the perturbator is centered
        pert_fcsd_squared_sum (real): sum of squared of the fcsd perturbator
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    pert_fcsd = Prop_pol(mf).pert_fcsd(atm_id)
    pert_fcsd_squared_sum_ = (pert_fcsd[0][1] * pert_fcsd[0][1]).sum()
    assert pert_fcsd_squared_sum - pert_fcsd_squared_sum_ < 1e-5

@pytest.mark.parametrize(
    " atm1_id, atm2_id, FC_response ", [([0], [1], [1.452389696720406e-08])]
)
def test_pp_fc(atm1_id, atm2_id, FC_response):
    """Test for the FC response

    Args:
        atm1_id (list): list with atm1 id
        atm2_id (list): list with atm2 id
        FC_response (real): value of the response
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    response = Prop_pol(mf).pp_fc(atm1_id, atm2_id)
    assert FC_response - response[0][0][0] < 1e-10

@pytest.mark.parametrize(
    " atm1_id, atm2_id, FCSD_response ", [([0], [1], [1.4523896967199533e-08])]
)
def test_pp_fcsd(atm1_id, atm2_id, FCSD_response):
    """Test for the FC+SD response

    Args:
        atm1_id (list): list with atm1 id
        atm2_id (list): list with atm2 id
        FCSD_response (real): value of the response
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    response = Prop_pol(mf).pp_fc(atm1_id, atm2_id)
    assert FCSD_response - response[0][0][0] < 1e-10

@pytest.mark.parametrize(
    " atm1_id, atm2_id, PSO_response ", [([0], [1], [1.8504633135969667e-08])]
)
def test_pp_PSO(atm1_id, atm2_id, PSO_response):
    """Test for the PSO response

    Args:
        atm1_id (list): list with atm1 id
        atm2_id (list): list with atm2 id
        PSO_response (real): value of the response
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    response = Prop_pol(mf).pp_pso(atm1_id, atm2_id)
    assert PSO_response - response[0][1][1] < 1e-10


@pytest.mark.parametrize(" atm1, atm2, FC_contribution ", [("H", "F", 208.10695206)])
def test_ssc(atm1, atm2, FC_contribution):
    """Test for the FC contribution

    Args:
        atm1_id (list): list with atm1 label
        atm2_id (list): list with atm2 label
        FC_contribution  (real): FC contribution to NR-SSC
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    ssc_fc = Prop_pol(mf).ssc(FC=True, atom1=atm1, atom2=atm2)
    assert ssc_fc - FC_contribution < 1e-5


@pytest.mark.parametrize(" atm1, atm2, PSO_contribution ", [("H", "F", 176.34185676)])
def test_ssc(atm1, atm2, PSO_contribution):
    """Test for PSO contribution

    Args:
        atm1_id (list): list with atm1 label
        atm2_id (list): list with atm2 label
        PSO_contribution  (real): PSO contribution to NR-SSC
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    ssc_pso = Prop_pol(mf).ssc(PSO=True, FC=False, atom1=atm1, atom2=atm2)
    assert ssc_pso - PSO_contribution < 1e-5

@pytest.mark.parametrize(" atm1, atm2, FCSD_contribution ", [("H", "F", 172.7419156)])
def test_ssc(atm1, atm2, FCSD_contribution):
    """Test for the FC+SD contribution

    Args:
        atm1_id (list): list with atm1 label
        atm2_id (list): list with atm2 label
        PSO_contribution  (real): PSO contribution to NR-SSC
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    ssc_pso = Prop_pol(mf).ssc(FCSD=True, FC=False, atom1=atm1, atom2=atm2)
    assert ssc_pso - FCSD_contribution < 1e-5
