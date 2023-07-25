import pytest
from pyscf import gto, scf
from pyppm.hrpa import HRPA

@pytest.mark.parametrize(" i, kappa2 ", [(1, 0.27746518421690647)])
def test_kappa(i, kappa2):
    """Test for Kappa function. It uses kappa^2 because kappa don't converge to a value 
    
    Args:
        i (int): 1 or 2
        kappa2 (real): kappa^2
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    kappa2_ = (pp.kappa(i)**2).sum()
    assert abs(kappa2 - kappa2_) < 1e-5

@pytest.mark.parametrize("a2", [2.2123806369774135])
def test_part_a2(a2):
    """Test for A(2) matrix

    Args:
        a2 (real): full A(2) matrix sum
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    a2_ = pp.part_a2.sum()
    assert abs(a2_ - a2) < 1e-5

@pytest.mark.parametrize("multiplicity, b2", [(1, 0.08999038175823586)])
def test_part_b2(multiplicity, b2):
    """Test for B(2) matrix

    Args:
        multiplicity (int): 1 for triplet responses
        b2 (real): sum of B(2)**2
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    b2_ = (pp.part_b2(multiplicity)**2).sum()
    assert abs(b2_ - b2) < 1e-5

@pytest.mark.parametrize("multiplicity, b2", [(0, 0.08055797334512305)])
def test_part_b2(multiplicity, b2):
    """Test for B(2) matrix

    Args:
        multiplicity (int): 0 for singlet responses
        b2 (real): sum of B(2)(0)**2
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    b2_ = (pp.part_b2(multiplicity)**2).sum()
    assert abs(b2_ - b2) < 1e-5

@pytest.mark.parametrize("s2", [-4.585927587663189])
def test_s2(s2):
    """Test for S(2) matrix

    Args:
        s2 (real): full S(2) matrix sum
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    s2_ = pp.S2.sum()
    assert abs(s2_ - s2) < 1e-5

@pytest.mark.parametrize("kappa2", [0.014538071653024773])
def test_kappa_2(kappa2):
    """Test for kappa_2 matrix

    Args:
        kappa2 (real): full kappa_2**2 matrix sum
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    kappa2_ = (pp.kappa_2**2).sum()
    assert abs(kappa2_ - kappa2) < 1e-5

@pytest.mark.parametrize("atmlst, correction", [([0], -5.421213435194959)])
def test_correction_pert(atmlst,correction):
    """Test for first correction to perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    correction_ = pp.correction_pert(atmlst).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atmlst, correction", [([0], 0.32347835225943505)])
def test_correction_pert_pso(atmlst,correction):
    """Test for first correction to PSO perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    correction_ = (pp.correction_pert_pso(atmlst)[0]**2).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atmlst, correction", [([0], -2.475781384137429)])
def test_correction_pert_2(atmlst,correction):
    """Test for second correction to perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    correction_ = pp.correction_pert_2(atmlst).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atmlst, correction", [([0], 1.192706252508168)])
def test_correction_pert_2_pso(atmlst,correction):
    """Test for second correction to PSO perturbation

    Args:
        atmlst (list): atom list
        correction (real): sum of correction
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    correction_ = (pp.correction_pert_2_pso(atmlst)[0]**2).sum()
    assert abs(correction_ - correction) < 1e-5

@pytest.mark.parametrize("atm1lst, atm2lst, fc_response", [([0], [1], -1.2714273e-08)])
def test_pp_ssc_fc_select(atm1lst,atm2lst,fc_response):
    """Test for FC Response at HRPA

    Args:
        atm1lst (list): atom list in which is centered first perturbator
        atm2lst (list): atom list in which is centered second perturbator
        correction (real): FC response value
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    fc_ = pp.pp_ssc_fc_select(atm1lst,atm2lst)[0][0][0]
    assert abs(fc_ - fc_response) < 1e-5

@pytest.mark.parametrize("atm1lst, atm2lst, pso_response", [([0], [1], -1.59064175697)])
def test_pp_ssc_pso_select(atm1lst,atm2lst,pso_response):
    """Test for PSO Response at HRPA

    Args:
        atm1lst (list): atom list in which is centered first perturbator
        atm2lst (list): atom list in which is centered second perturbator
        correction (real): PSO response value
    """
    mol = gto.M(atom='''
    F     0.0000000000    0.0000000000     0.1319629808
    H     0.0000000000    0.0000000000    -1.6902522555
    ''', basis='3-21g', unit='angstrom')

    mf = mol.RHF().run()
    pp = HRPA(mf)
    pso_ = pp.pp_ssc_pso_select(atm1lst,atm2lst)[0][0][0]
    assert abs(pso_ - pso_response) < 1e-5