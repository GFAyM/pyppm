import pytest
from pyscf import gto, scf
from pyppm.rpa import RPA

@pytest.mark.parametrize("eri_mo_2_sum", [(230.43674870949738)])
def test_eri_mo(eri_mo_2_sum):
    """test for eri_mo property

    Args:
        eri_mo_2_sum (real): sum of eri_mo squared
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()
    rpa_obj = RPA(mf=mf)
    eri_mo = rpa_obj.eri_mo()
    assert abs(eri_mo_2_sum - (eri_mo**2).sum()) < 1e-5


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
    m = RPA(mf).M(triplet=Triplet)
    assert M_trace - m.trace() < 1e-5

@pytest.mark.parametrize('triplet, q_2_sum', [(True, 46.95770261152346)])
def test_Communicator(triplet, q_2_sum):
    """test for communicator function, triplet

    Args:
        triplet (bool): triplet
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()

    rpa_obj = RPA(mf=mf)
    q = rpa_obj.Communicator(triplet=triplet)
    assert abs(q_2_sum - (q**2).sum()) < 1e-5

@pytest.mark.parametrize('triplet, q_2_sum', [(False, 34.06845366122354)])
def test_Communicator(triplet, q_2_sum):
    """test for communicator function, singlet

    Args:
        triplet (bool): triplet
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()

    rpa_obj = RPA(mf=mf)
    q = rpa_obj.Communicator(triplet=triplet)
    assert abs(q_2_sum - (q**2).sum()) < 1e-5

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
    prop = RPA(mf)
    id = prop.obtain_atom_order(Element)
    assert id == I


@pytest.mark.parametrize(" atm_id, pert_fc_sum ", [([0], [0.17070183232781733])])
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
    pert = RPA(mf).pert_fc(atm_id)
    assert pert_fc_sum - pert[0].sum() < 1e-5


@pytest.mark.parametrize(" atm_id, pert_pso_squared_sum ", [([0], [0.5583537])])
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
    pp = RPA(mf)
    pert_pso = pp.pert_pso(atm_id)
    nocc = pp.nocc
    pert_pso_squared_sum_ = (pert_pso[2][:nocc,nocc:]**2).sum()
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
    fcsd_integrals_ = RPA(mf).get_integrals_fcsd(atm_id)
    assert fcsd_integrals - fcsd_integrals_.sum() < 1e-2

@pytest.mark.parametrize(" atm_id, pert_fcsd_squared_sum ", [([0], [0.1008467])])
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
    pert_fcsd = RPA(mf).pert_fcsd(atm_id)
    pert_fcsd_squared_sum_ = (pert_fcsd[0][1]).sum()
    assert abs(pert_fcsd_squared_sum - pert_fcsd_squared_sum_) < 1e-5

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
    response = RPA(mf).pp_fc(atm1_id, atm2_id)
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
    response = RPA(mf).pp_fc(atm1_id, atm2_id)
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
    response = RPA(mf).pp_pso(atm1_id, atm2_id)
    assert PSO_response - response[0][1][1] < 1e-10


@pytest.mark.parametrize("atm1, atm2, FC_contribution ", [('H', 'F', 208.10695206)])
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
    ssc_fc = RPA(mf).ssc(FC=True, atom1=atm1, atom2=atm2)
    assert abs(ssc_fc - FC_contribution) < 1e-4


@pytest.mark.parametrize("atm1, atm2, PSO_contribution ", [("H", "F", 176.34185676)])
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
    ssc_pso = RPA(mf).ssc(PSO=True, FC=False, atom1=atm1, atom2=atm2)
    assert ssc_pso - PSO_contribution < 1e-5

@pytest.mark.parametrize("atm1, atm2, FCSD_contribution", [("H", "F", 172.7419156)])
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
    ssc_pso = RPA(mf).ssc(FCSD=True, atom1=atm1, atom2=atm2)
    assert abs(ssc_pso - FCSD_contribution) < 1e-4

@pytest.mark.parametrize('atm1lst, atm2lst, fc, h1, m2, h2', 
                         [([0], [1], True, 
                           48.14609886902724, 
                           11352.983035322855, 
                           316760.53893761913)])
def test_elements(atm1lst, atm2lst, fc, h1, m2, h2):
    """Test for element function, fc mechanism

    Args:
        atm1lst (list): atm1 list
        atm2lst (list): atm2 list
        fc (bool): Mechanism
        h1 (numpy.ndarray): perturbator centered in atm1, the sum
        m2 (numpy.ndarray): principal propagator inverse square, the sum
        h2 (numpy.ndarray): perturbator centered in atm1, the sum
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)
    mf.kernel()

    hrpa_obj = RPA(mf=mf)
    h1_, m_, h2_ = hrpa_obj.elements(atm1lst=atm1lst, atm2lst=atm2lst, 
                                  FC=fc)
    assert abs((h1_**2).sum() - h1) < 1e-5
    assert abs((h2_**2).sum() - h2) < 1e-5
    assert abs((m_**2).sum() - m2) < 1e-5

@pytest.mark.parametrize('atm1lst, atm2lst, fcsd, h1, m2, h2', 
                         [([0], [1], True, 
                           37.10897944570908,
                           11352.983035322852,
                           238073.57886809413)])
def test_elements(atm1lst, atm2lst, fcsd, h1, m2, h2):
    """Test for element function, fcsd mechanism

    Args:
        atm1lst (list): atm1 list
        atm2lst (list): atm2 list
        fc (bool): Mechanism
        h1 (numpy.ndarray): perturbator centered in atm1, the sum
        m2 (numpy.ndarray): principal propagator inverse square, the sum
        h2 (numpy.ndarray): perturbator centered in atm1, the sum
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)

    mf.kernel()

    hrpa_obj = RPA(mf=mf)
    h1_, m_, h2_ = hrpa_obj.elements(atm1lst=atm1lst, atm2lst=atm2lst, 
                                  FCSD=fcsd)
    assert abs((h1_**2).sum() - h1) < 1e-5
    assert abs((h2_**2).sum() - h2) < 1e-5
    assert abs((m_**2).sum() - m2) < 1e-5

@pytest.mark.parametrize('atm1lst, atm2lst, pso, h1, m2, h2', 
                         [([0], [1], True, 
                           4.724843804357335,
                           11418.512146853163,
                           1341.4330674042815)])
def test_elements(atm1lst, atm2lst, pso, h1, m2, h2):
    """Test for element function, pso mechanism

    Args:
        atm1lst (list): atm1 list
        atm2lst (list): atm2 list
        fc (bool): Mechanism
        h1 (numpy.ndarray): perturbator centered in atm1, the sum
        m2 (numpy.ndarray): principal propagator inverse square, the sum
        h2 (numpy.ndarray): perturbator centered in atm1, the sum
    """
    HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
    mf = scf.RHF(HF_mol)

    mf.kernel()

    hrpa_obj = RPA(mf=mf)
    h1_, m_, h2_ = hrpa_obj.elements(atm1lst=atm1lst, atm2lst=atm2lst, 
                                  PSO=pso)
    assert abs((h1_**2).sum() - h1) < 1e-3
    assert abs((h2_**2).sum() - h2) < 1e-3
    assert abs((m_**2).sum() - m2) < 1e-3