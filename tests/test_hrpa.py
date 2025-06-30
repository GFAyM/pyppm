import pytest
from pyscf import gto, scf
from pyppm.hrpa import HRPA

@pytest.fixture(scope="module")
def hf_data():
    """
    Fixture that returns the molecule and RHF wavefunction for the HF molecule.
    The SCF result is saved to a checkpoint file.
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')
    mf = scf.RHF(mol)
    mf.chkfile = 'HF_test_hrpa.chk'
    mf.kernel()
    return mol, mf.chkfile


@pytest.mark.parametrize(" i, kappa2 ", [(1, 0.27746518421690647)])
def test_kappa(i, kappa2, hf_data):
    """Test for Kappa function. It uses kappa^2 because kappa don't converge to a value

    Args:
        i (int): 1 or 2
        kappa2 (real): kappa^2
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=True)
    kappa2_ = (pp.kappa(i)**2).sum()
    assert abs(kappa2 - kappa2_) < 1e-5

@pytest.mark.parametrize("a2", [2.2123806369774135])
def test_part_a2(a2, hf_data):
    """Test for A(2) matrix

    Args:
        a2 (real): full A(2) matrix sum
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    a2_ = pp.part_a2.sum()
    assert abs(a2_ - a2) < 1e-5

@pytest.mark.parametrize("multiplicity, b2", [(1, 0.08999038175823586)])
def test_part_b2(multiplicity, b2, hf_data):
    """Test for B(2) matrix

    Args:
        multiplicity (int): 1 for triplet responses
        b2 (real): sum of B(2)**2
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    b2_ = (pp.part_b2(multiplicity)**2).sum()
    assert abs(b2_ - b2) < 1e-5


@pytest.mark.parametrize("multiplicity, b2", [(0, 0.08055797334512305)])
def test_part_b2(multiplicity, b2, hf_data):
    """Test for B(2) matrix
    
    Args:
        multiplicity (int): 0 for singlet responses
        b2 (real): sum of B(2)(0)**2
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    b2_ = (pp.part_b2(multiplicity)**2).sum()
    assert abs(b2_ - b2) < 1e-5

@pytest.mark.parametrize("s2", [-4.585927587663189])
def test_s2(s2, hf_data):
    """Test for S(2) matrix

    Args:
        s2 (real): full S(2) matrix sum
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    s2_ = pp.S2.sum()
    assert abs(s2_ - s2) < 1e-5

@pytest.mark.parametrize("kappa2", [0.014538071653024773])
def test_kappa_2(kappa2, hf_data):
    """Test for kappa_2 matrix

    Args:
        kappa2 (real): full kappa_2**2 matrix sum
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    kappa2_ = (pp.kappa_2**2).sum()
    assert abs(kappa2_ - kappa2) < 1e-5

@pytest.mark.parametrize("atmlst, pert, correction", [([0], 'FC', -5.421213435194959)])
def test_correction_pert(atmlst,pert, correction, hf_data):
    """Test for first correction to perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    if pert == 'FC':
        correction_ = pp.correction_pert(atmlst=atmlst,FC=True).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atmlst, pert, correction", [([0],'PSO', 0.32347835225943505)])
def test_correction_pert(atmlst, pert, correction, hf_data):
    """Test for first correction to PSO perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    if pert =='PSO':
        correction_ = (pp.correction_pert(atmlst=atmlst, PSO=True)[0]**2).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atmlst, pert, correction", [([0],'FCSD', 10.27766223804352)])
def test_correction_pert(atmlst, pert, correction, hf_data):
    """Test for first correction to FCSD perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    if pert =='FCSD':
        correction_ = (pp.correction_pert(atmlst=atmlst, FCSD=True)[0]**2).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atmlst, pert, correction", [([0], 'FC', -2.475781384137429)])
def test_correction_pert_2(atmlst, pert, correction, hf_data):
    """Test for second correction to FC perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    if pert:
        correction_ = pp.correction_pert_2(atmlst=atmlst, FC=True).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atmlst, pert, correction", [([0], 'PSO', 1.192706252508168)])
def test_correction_pert_2(atmlst, pert, correction, hf_data):
    """Test for second correction to PSO perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    if pert:
        correction_ = pp.correction_pert_2(atmlst=atmlst, PSO=True).sum()
    assert abs(correction_ - correction) < 1e-5
    
@pytest.mark.parametrize("atmlst, pert, correction", [([0], 'FCSD', 3.942935194771017)])
def test_correction_pert_2(atmlst, pert, correction, hf_data):
    """Test for second correction to FCSD perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    if pert == 'FCSD':
        correction_ = (pp.correction_pert_2(atmlst=atmlst, FCSD=True)[0]**2).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atm1lst, atm2lst, fc_response", [([0], [1], -1.2714273e-08)])
def test_pp_ssc_fc(atm1lst,atm2lst,fc_response, hf_data):
    """Test for FC Response at HRPA

    Args:
        atm1lst (list): atom list in which is centered first perturbator
        atm2lst (list): atom list in which is centered second perturbator
        correction (real): FC response value
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    fc_ = pp.pp_ssc_fc(atm1lst,atm2lst)[0][0][0]
    assert abs(fc_ - fc_response) < 1e-5

@pytest.mark.parametrize("atm1lst, atm2lst, pso_response", [([0], [1], 4.5106034805326434e-09)])
def test_pp_ssc_pso(atm1lst,atm2lst,pso_response, hf_data):
    """Test for PSO Response at HRPA

    Args:
        atm1lst (list): atom list in which is centered first perturbator
        atm2lst (list): atom list in which is centered second perturbator
        correction (real): PSO response value
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    pso_ = pp.pp_ssc_pso(atm1lst,atm2lst)[0][0][0]
    assert abs(pso_ - pso_response) < 1e-5

@pytest.mark.parametrize("atm1lst, atm2lst, fcsd_response", [([0], [1], 1.3137210541617598e-07)])
def test_pp_ssc_fcsd(atm1lst,atm2lst,fcsd_response, hf_data):
    """Test for DC+SD Response at HRPA

    Args:
        atm1lst (list): atom list in which is centered first perturbator
        atm2lst (list): atom list in which is centered second perturbator
        correction (real): PSO response value
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    fcsd = pp.pp_ssc_fcsd(atm1lst,atm2lst)[0][0][0]
    assert abs(fcsd - fcsd_response) < 1e-5

@pytest.mark.parametrize('atom1, atom2, ssc, fc', [('F', 'H', -182.17405198617374, True)])
def test_ssc(atom1, atom2, ssc, fc, hf_data):
    """Test por fc-ssc

    Args:
        atom1 (str): atom1 string
        atom2 (str): atom2 string
        ssc (real): ssc value
        fc (bool): fc mechanism
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    ssc_fc = pp.ssc(atom1=atom1, atom2=atom2, FC=fc)
    assert abs(ssc_fc - ssc) < 1e-5

@pytest.mark.parametrize('atom1, atom2, ssc, fcsd', [('F', 'H', -121.42222653182456, True)])
def test_ssc(atom1, atom2, ssc, fcsd, hf_data):
    """Test por fc+sd-ssc

    Args:
        atom1 (str): atom1 string
        atom2 (str): atom2 string
        ssc (real): ssc value
        fcsd (bool): fc mechanism
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    ssc_fc = pp.ssc(atom1=atom1, atom2=atom2, FCSD=fcsd)
    assert abs(ssc_fc - ssc) < 1e-5

@pytest.mark.parametrize('atom1, atom2, ssc, pso', [('F', 'H', 43.08679427289697, True)])
def test_ssc(atom1, atom2, ssc, pso, hf_data):
    """Test por fc+sd-ssc

    Args:
        atom1 (str): atom1 string
        atom2 (str): atom2 string
        ssc (real): ssc value
        pso (bool): pso mechanism
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    ssc_pso = pp.ssc(atom1=atom1, atom2=atom2, PSO=pso)
    assert abs(ssc_pso - ssc) < 1e-3

@pytest.mark.parametrize('atm1lst, atm2lst, fc, h1, m2, h2', 
                         [([0], [1], True, -406.3275653,  4365.3773397, -5.6440132)])
def test_elements(atm1lst, atm2lst, fc, h1, m2, h2, hf_data):
    """Test for element function

    Args:
        atm1lst (list): atm1 list
        atm2lst (list): atm2 list
        fc (bool): Mechanism
        h1 (numpy.ndarray): perturbator centered in atm1, the sum
        m2 (numpy.ndarray): principal propagator inverse square, the sum
        h2 (numpy.ndarray): perturbator centered in atm1, the sum
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    h1_, m_, h2_ = pp.elements(atm1lst=atm1lst, atom2lst=atm2lst, 
                                  FC=fc)
    assert abs(h1_.sum() - h1) < 1e-5
    assert abs(h2_.sum() - h2) < 1e-5
    assert abs((m_**2).sum() - m2) < 1e-5

@pytest.mark.parametrize('atm1lst, atm2lst, fcsd, h1, m2, h2', 
                         [([0], [1], True, 
                           216010.98419301497, 4365.377339775919, 18.45493906561035)])
def test_elements(atm1lst, atm2lst, fcsd, h1, m2, h2, hf_data):
    """Test for element function, for FC+SD mechanism

    Args:
        atm1lst (list): atm1 list
        atm2lst (list): atm2 list
        fcsd (bool): Mechanism
        h1 (numpy.ndarray): perturbator centered in atm1, the sum
        m2 (numpy.ndarray): principal propagator inverse square, the sum
        h2 (numpy.ndarray): perturbator centered in atm1, the sum
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    h1_, m_, h2_ = pp.elements(atm1lst=atm1lst, atom2lst=atm2lst, 
                                  FCSD=fcsd)
    assert abs((h1_**2).sum() - h1) < 1e-5
    assert abs((h2_**2).sum() - h2) < 1e-5
    assert abs((m_**2).sum() - m2) < 1e-5

@pytest.mark.parametrize('atm1lst, atm2lst, pso, h1, m2, h2', 
                         [([0], [1], True, 
                           1397.492728818434, 
                           4413.033196930066,
                           0.20265197381571692)])
def test_elements(atm1lst, atm2lst, pso, h1, m2, h2, hf_data):
    """Test for element function, for pso mechanism

    Args:
        atm1lst (list): atm1 list
        atm2lst (list): atm2 list
        fcsd (bool): Mechanism
        h1 (numpy.ndarray): perturbator centered in atm1, the sum
        m2 (numpy.ndarray): principal propagator inverse square, the sum
        h2 (numpy.ndarray): perturbator centered in atm1, the sum
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    h1_, m_, h2_ = pp.elements(atm1lst=atm1lst, atom2lst=atm2lst, 
                                  PSO=pso)
    assert abs((h1_**2).sum() - h1) < 1e-3
    assert abs((h2_**2).sum() - h2) < 1e-3
    assert abs((m_**2).sum() - m2) < 1e-3

@pytest.mark.parametrize('triplet, q_2_sum', [(True, 28.62154921395211)])
def test_Communicator(triplet, q_2_sum, hf_data):
    """test for communicator function, triplet

    Args:
        triplet (bool): triplet
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    q = pp.Communicator(triplet=triplet)
    assert abs(q_2_sum - (q**2).sum()) < 1e-5

@pytest.mark.parametrize('triplet, q_2_sum', [(False, 22.277567990815314)])
def test_Communicator(triplet, q_2_sum, hf_data):
    """test for communicator function, singlet

    Args:
        triplet (bool): triplet
    """
    mol, chkfile = hf_data
    pp = HRPA(mol=mol, chkfile=chkfile, mole_name=None, calc_int=False)
    q = pp.Communicator(triplet=triplet)
    assert abs(q_2_sum - (q**2).sum()) < 1e-5
