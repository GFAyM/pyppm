import pytest
from pyscf import scf
from pyppm.help_functions import extra_functions
from pyppm.loc import Loc
import os

main_directory=os.path.realpath(os.path.dirname(__file__))+'/../'

@pytest.mark.parametrize("level",[('HRPA')])
def test__attrs_post_init__(level):
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = f'{level}')

    with pytest.raises(Exception) as exc_info:    
        loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = f'{level}')
        assert str(exc_info.value
                   ) == "SOPPA or other method are not avaible yet. Only RPA and HRPA"

@pytest.mark.parametrize("c_occ, v, c_vir",[(4.999999999999936, 69.99999999999824, 13.999999999999826)])
def test_inv_mat(c_occ, v, c_vir):
    """testing inv_mat property

    Args:
        c_occ (real): sum of c_occ square
        v (real): sum of v square
        c_vir (real): sum of c_vir square
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = 'RPA')
    c_occ_, v_, c_vir_ = loc_obj.inv_mat
    c_occ_ = (c_occ_**2).sum()
    c_vir_ = (c_vir_**2).sum()
    v_ = (v_**2).sum()
    assert abs(c_occ_ - c_occ) < 1e-5
    assert abs(c_vir_ - c_vir) < 1e-5
    assert abs(v_ - v) < 1e-5    



@pytest.mark.parametrize("h1, p, h2, fc, corr", [(468.19341140496437, 
                                            -70.94801320523182,
                                            -0.6653255733645091, 
                                            True, 'RPA')])
def test_pp(h1, p, h2, fc, corr):
    """testing pp function.

    Args:
        h1 (real): sum of h1
        p (real): sum of p
        h2 (real): sum of h2
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = str(corr))
    h1_loc, p_loc, h2_loc = loc_obj.pp(atom1='F1', atom2='H2', FC=fc)
    assert abs(h1_loc.sum() - h1) < 1e-5
    assert abs(h2_loc.sum() - h2) < 1e-5
    assert abs(p_loc.sum() - p) < 1e-5    

@pytest.mark.parametrize("h1, p, h2, fc, corr", [(463.07157247455143, 
                                                  -76.45375105907738, 
                                                  -0.5742839919486595,
                                                  True, 'HRPA')])
def test_pp(h1, p, h2, fc, corr):
    """testing pp function.

    Args:
        h1 (real): sum of h1
        p (real): sum of p
        h2 (real): sum of h2
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = str(corr))
    h1_loc, p_loc, h2_loc = loc_obj.pp(atom1='F1', atom2='H2', FC=fc)
    assert abs(h1_loc.sum() - h1) < 1e-5
    assert abs(h2_loc.sum() - h2) < 1e-5
    assert abs(p_loc.sum() - p) < 1e-5   

@pytest.mark.parametrize("h1, p, h2, pso, corr", [(0.025459807807276746, 
                                                   -59.235248661161435, 
                                                   -0.32408736243866565,
                                                   True, 'RPA')])
def test_pp(h1, p, h2, pso, corr):
    """testing pp function.

    Args:
        h1 (real): sum of h1
        p (real): sum of p
        h2 (real): sum of h2
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = str(corr))
    h1_loc, p_loc, h2_loc = loc_obj.pp(atom1='F1', atom2='H2', PSO=pso)
    assert abs(h1_loc.sum() - h1) < 1e-5
    assert abs(h2_loc.sum() - h2) < 1e-5
    assert abs(p_loc.sum() - p) < 1e-5   

@pytest.mark.parametrize("h1, p, h2, fcsd, corr", [(691.391722744511,
                                                    -70.94801320520783, 
                                                    -1.0177345128964537,
                                                    True, 'RPA')])
def test_pp(h1, p, h2, fcsd, corr):
    """testing pp function.

    Args:
        h1 (real): sum of h1
        p (real): sum of p
        h2 (real): sum of h2
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = str(corr))
    h1_loc, p_loc, h2_loc = loc_obj.pp(atom1='F1', atom2='H2', FCSD=fcsd)
    assert abs(h1_loc.sum() - h1) < 1e-5
    assert abs(h2_loc.sum() - h2) < 1e-5
    assert abs(p_loc.sum() - p) < 1e-5 

@pytest.mark.parametrize("j_pso, pso, hrpa", [(40.2215594660588, 
                                               True,
                                               'HRPA')])
def test_ssc(j_pso, pso, hrpa):
    """testing ssc function for pso and hrpa.

    Args:
        j_pso (real): ssc for pso 
        pso (bool): PSO mechanism
        HRPA (str): level of approach
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = hrpa)
    pso_ = loc_obj.ssc(atom1='F1', atom2='H2', PSO=pso)
    assert abs(pso_ - j_pso) < 1e-5

@pytest.mark.parametrize("j_pso, pso, rpa", [(98.95020755285924, 
                                               True,
                                               'RPA')])
def test_ssc(j_pso, pso, rpa):
    """testing ssc function for pso and hrpa.

    Args:
        j_pso (real): ssc for pso 
        pso (bool): PSO mechanism
        HRPA (str): level of approach
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = rpa)
    pso_ = loc_obj.ssc(atom1='F1', atom2='H2', PSO=pso)
    assert abs(pso_ - j_pso) < 1e-5

@pytest.mark.parametrize("j_fcsd, fcsd, rpa", [(-102.22320818916083, 
                                               True,
                                               'RPA')])
def test_ssc(j_fcsd, fcsd, rpa):
    """testing ssc function for pso and hrpa.

    Args:
        j_pso (real): ssc for pso 
        pso (bool): PSO mechanism
        HRPA (str): level of approach
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = rpa)
    fcsd_ = loc_obj.ssc(atom1='F1', atom2='H2', FCSD=fcsd)
    assert abs(fcsd_ - j_fcsd) < 1e-3


@pytest.mark.parametrize("fcsd_iajb, FCSD, corr", 
                            [(-102.22320818915543, True,
                              'RPA')])
def test_ssc_pathways(fcsd, FCSD, corr):
    """testing ssc_pathways function for fcsd and hrpa 

    Args:
        fcsd (real): ssc for fc+sd mechanisms
        FCSD (bool): fcsd mechansm
        corr (str): correlation level
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = corr)
    occ = loc_obj.occidx
    fcsd_ij = loc_obj.ssc_pathways(atom1='F1', atom2='H2',
                                    FCSD=FCSD,
                                    occ_atom1=occ,
                                    occ_atom2=occ)
    assert abs(fcsd_ij - fcsd) < 1e-5

@pytest.mark.parametrize("PSO, corr", 
                            [(True, 'RPA')])
def test_ssc_pathways(PSO, corr):
    """testing ssc_pathways function for fcsd and hrpa 

    Args:
        fcsd (real): ssc for fc+sd mechanisms
        FCSD (bool): fcsd mechansm
        corr (str): correlation level
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = corr)
    ssc = loc_obj.ssc(atom1='F1', atom2='H2', PSO=PSO)
    occ = loc_obj.occidx
    vir = loc_obj.viridx
    ssc_iajb = loc_obj.ssc_pathways(atom1='F1', atom2='H2',
                                    PSO=PSO,
                                    occ_atom1=occ,
                                    occ_atom2=occ)
    assert abs(ssc - ssc_iajb) < 1e-5

@pytest.mark.parametrize("fcsd_iajb, FCSD,  occ1, vir1, occ2, vir2", 
                            [(0.002397787579269773, True,
                              [3,4], [10,12,13],
                              [0,1], [9,11,14])])
def test_ssc_pathways(fcsd_iajb, FCSD, occ1, vir1, occ2, vir2):
    """testing ssc_pathways function for fcsd and hrpa 

    Args:
        fcsd_iajb (real): ssc for fc+sd mechanisms, for iajb pathways
        occ1 (_type_): set of ocuppied orbitals for atom1 nuclei
        vir1 (_type_): set of virtual orbitals for atom1 nuclei
        occ2 (_type_): set of ocuppied orbitals for atom2 nuclei
        vir2 (_type_): set of virtual orbitals for atom2 nuclei
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = 'RPA')
    fcsd_iajb_ = loc_obj.ssc_pathways(atom1='F1', atom2='H2',
                                    FCSD=FCSD,
                                    occ_atom1=occ1, vir_atom1=vir1,
                                    occ_atom2=occ2, vir_atom2=vir2)
    assert abs(fcsd_iajb_ - fcsd_iajb) < 1e-5
