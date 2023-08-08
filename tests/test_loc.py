import pytest
from pyscf import scf
from pyppm.help_functions import extra_functions
from pyppm.loc import Loc
import os

main_directory=os.path.realpath(os.path.dirname(__file__))+'/../'

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


@pytest.mark.parametrize("h1, m, h2, fc", [(468.19341140496437, 
                                            525.5107556687838,
                                            -0.6653255733645091, 
                                            True)])
def test_pp(h1, m, h2, fc):
    """testing pp function.

    Args:
        h1 (real): sum of h1
        m (real): sum of m
        h2 (real): sum of h2
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = 'RPA')
    h1_loc, m_loc, h2_loc = loc_obj.pp(atom1='F1', atom2='H2', FC=fc)
    assert abs(h1_loc.sum() - h1) < 1e-5
    assert abs(h2_loc.sum() - h2) < 1e-5
    assert abs(m_loc.sum() - m) < 1e-5    

@pytest.mark.parametrize("h1, m, h2, fc", [(463.07156524601197,
                                            524.6033771786385,
                                            -0.5742838151231682, 
                                            True)])
def test_pp(h1, m, h2, fc):
    """testing pp function.

    Args:
        h1 (real): sum of h1
        m (real): sum of m
        h2 (real): sum of h2
    """

    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = 'HRPA')
    h1_loc, m_loc, h2_loc = loc_obj.pp(atom1='F1', atom2='H2', FC=fc)
    assert abs(h1_loc.sum() - h1) < 1e-5
    assert abs(h2_loc.sum() - h2) < 1e-5
    assert abs(m_loc.sum() - m) < 1e-5   

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

@pytest.mark.parametrize("fcsd_iajb, occ1, vir1, occ2, vir2", 
                            [(-0.5314627516370206, 
                              [3,4], [10,12,13],
                              [0,1], [9,11,14])])
def test_ssc_pathways(fcsd_iajb, occ1, vir1, occ2, vir2):
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
    loc_obj = Loc(mf=mf, mo_coeff_loc = mo_coeff, elec_corr = 'HRPA')
    h1_loc, m_loc, h2_loc = loc_obj.pp(atom1='F1', atom2='H2', FCSD=True)
    fcsd_iajb_ = loc_obj.ssc_pathways(atom1='F1', atom2='H2',
                           h1 = h1_loc, m = m_loc, h2 = h2_loc,
                           occ_atom1=occ1, vir_atom1=vir1,
                           occ_atom2=occ2, vir_atom2=vir2)
    assert abs(fcsd_iajb_ - fcsd_iajb) < 1e-5
