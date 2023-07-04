import pytest
from pyppm.help_functions import extra_functions
from pyppm.ssc_cloppa import Cloppa
import os
import numpy as np

main_directory=os.path.realpath(os.path.dirname(__file__))+'/../'

@pytest.mark.parametrize("atom1, atom2, value", 
                         [('H3', 'H7', -1.00180548e-07)])

def test_ssc_pathway(atom1,atom2,value):
    """SSC pathway test

    Args:
        atom1 (str): atom1 name
        atom2 (str): atom2 name
        value (int): value of the pathway
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    ssc_path = cloppa_obj.ssc_pathway(atom1=atom1, atom2=atom2, occ_atom1=3, 
                           vir_atom1=16, occ_atom2=4, vir_atom2=19, FC=False
                           ,PSO=True, FCSD=False)
    assert abs(value - ssc_path) < 1e-10

@pytest.mark.parametrize("atom1, atom2, value", 
                         [('H3', 'H7', 12.4891868569)])

def test_ssc_pathway(atom1,atom2,value):
    """SSC pathway test for all coupling pathways

    Args:
        atom1 (str): atom1 name
        atom2 (str): atom2 name
        value (int): value of ssc mechanism with all pathways
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    ssc = cloppa_obj.ssc_pathway(atom1=atom1, atom2=atom2,  
                            FC=True, PSO=False, FCSD=False, all_pathways=True)
    assert abs(value - ssc) < 1e-5

@pytest.mark.parametrize("atom1, atom2, value_p1, value_m, value_p2", 
                         [('H3', 'H7', 5.099783349336234e-06, 
                            1.9480504868517283e-05, 0.02693586737020855)])

def test_pathway_elements(atom1,atom2, value_p1, value_m, value_p2):
    """SSC pathway 

    Args:
        atom1 (str): atom1 name
        atom2 (str): atom2 name
        value_p1 (real): P$_{ia}$(atom1) perturbator centered in the atom1 
        value_m (real): M$_{ia,jb}$ Principal propagator for a definite pathway
        value_p2 (real): P$_{jb}$(atom2) perturbator centered in the atom2
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    p1, m, p2 = cloppa_obj.pathway_elements(FC=False, FCSD=True, PSO=False,
                        atom1='H3', occ_atom1=1, vir_atom1=10,
                        atom2='H7', occ_atom2=2, vir_atom2=12)
    assert abs(value_p1-p1) < 1e-10
    assert abs(value_p2-p2) < 1e-10
    assert abs(value_m - m) < 1e-10

@pytest.mark.parametrize("Element, I", [("H3", 2)])
def test_obtain_atom_order(Element, I):
    """Test for the function that gives the id of the element chosen

    Args:
        Element (str): Element
        ID (int): id number of the atom in the molecule
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    id = cloppa_obj.obtain_atom_order(Element)
    assert id == I

@pytest.mark.parametrize("n_atom1, n_atom2, value", 
                         [([2], [6], 8.204503729511254e-10)])

def test_pp_fc_pathways(n_atom1,n_atom2,value):
    """FC response test for all coupling pathways

    Args:
        atom1 (str): atom1 name
        atom2 (str): atom2 name
        value (int): value of ssc mechanism with all pathways
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    pp_fc = cloppa_obj.pp_fc_pathways(n_atom1=n_atom1, n_atom2=n_atom2, 
                    all_pathways=True,
                    occ_atom1=None, vir_atom1=None, occ_atom2=None,
                    vir_atom2=None, elements=False, 
                    princ_prop=np.full((2,2), None))
    assert abs(value - pp_fc[0][0][0]) < 1e-14


@pytest.mark.parametrize("n_atom1, n_atom2, value", 
                         [([2], [6], 7.918433061444212e-10)])

def test_pp_fcsd_pathways(n_atom1,n_atom2,value):
    """FCSD response test for all coupling pathways

    Args:
        atom1 (str): atom1 name
        atom2 (str): atom2 name
        value (int): value of firts fcsd response with all pathways
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    pp_fcsd = cloppa_obj.pp_fcsd_pathways(n_atom1=n_atom1, n_atom2=n_atom2, 
                    all_pathways=True,
                    occ_atom1=None, vir_atom1=None, occ_atom2=None,
                    vir_atom2=None, elements=False, 
                    princ_prop=np.full((2,2), None))
    assert abs(value - pp_fcsd[0][0][0]) < 1e-14

@pytest.mark.parametrize("n_atom1, n_atom2, value", 
                         [([2], [6], -1.1600129557380199e-10)])
def test_pp_pso_pathways(n_atom1,n_atom2,value):
    """PSO response test for all coupling pathways

    Args:
        atom1 (str): atom1 name
        atom2 (str): atom2 name
        value (int): value of firts PSO response with all pathways
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    pp_pso = cloppa_obj.pp_pso_pathways(n_atom1=n_atom1, n_atom2=n_atom2, 
                    all_pathways=True,
                    occ_atom1=None, vir_atom1=None, occ_atom2=None,
                    vir_atom2=None, elements=False, 
                    princ_prop=np.full((2,2), None))
    assert abs(value - pp_pso[0][0][0]) < 1e-14

@pytest.mark.parametrize(" atm_id, pert_fcsd_squared_sum ", [([1], [31.832289675231266])])
def test_pert_fcsd(atm_id, pert_fcsd_squared_sum):
    """Test for the fcsd perturbator
    
    Args:
        atm_id (list): List with the id of the atom in wich the perturbator is centered
        pert_fcsd_squared_sum (real): fcsd perturbator sum
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    pert = cloppa_obj.pert_fcsd(atm_id)
    #pert_fcsd_squared_sum_ = (pert_fcsd[0][1] * pert_fcsd[0][1]).sum()
    assert abs(pert_fcsd_squared_sum - pert[0].sum()) < 1e-5

@pytest.mark.parametrize(" atm_id, pert_fc_sum ", [([1], [354.24817683077976])])
def test_pert_fc(atm_id, pert_fc_sum):
    """
    Test for Perturbator using localized molecular orbitals
    it use the sum

    Args:
        atm_id (list): list with the id of the atom in wich the perturbator is centered
        pert_fc_sum (real) : perturbator sum
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    pert = cloppa_obj.pert_fc(atm_id)
    assert abs(pert_fc_sum - pert[0].sum()) < 1e-5

@pytest.mark.parametrize("atm_id, pert_pso", [([1], [-0.0038919427255503375])])
def test_pert_PSO(atm_id, pert_pso):
    """
    Test for PSO Perturbator using localized molecular orbitals
    it use the sum

    Args:
        atm_id (list): list with the id of the atom in wich the perturbator is centered
        pert_fc_sum (real) : perturbator sum
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    pert = cloppa_obj.pert_pso(atm_id)
    assert abs(pert_pso - pert[0].sum()) < 1e-5

@pytest.mark.parametrize(" atm_id, fcsd_integrals ", [(1, [220.39783593117323])])
def test_get_integrals_fcsd(atm_id, fcsd_integrals):
    """Test for the fcsd integrals using localized molecular orbitals
    Args:
        atm_id (int): atom id in wich the integrals are centered
        pert_fcsd_squared_sum (real): sum of squared of the fcsd perturbator
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    fcsd_integrals_ = cloppa_obj._get_integrals_fcsd(atm_id)
    assert abs(fcsd_integrals - fcsd_integrals_[0].sum()) < 1e-5

@pytest.mark.parametrize(" Triplet, M_trace ", [(True, [1825.752538356537])])
def test_M(Triplet, M_trace):
    """
    Test for Inverse of the Principal Propagator Matrix using Localized 
    molecular orbitals
    it use the trace

    Args:
        Triplet (boolean): if the response is triplet or singlet
        M_trace (real) : value of the M trace
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    m = cloppa_obj.M(triplet=Triplet)
    assert abs(M_trace - m.trace()) < 1e-5



@pytest.mark.parametrize(" M_trace ", [([1837.9110906360743])])
def test_M(M_trace):
    """
    Test for Inverse of the Principal Propagator Matrix using Localized 
    molecular orbitals
    it use the trace

    Args:
        Triplet (boolean): if the response is triplet or singlet
        M_trace (real) : value of the M trace
    """
    molden= main_directory + "tests/C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(
                            molden_file=molden).extraer_coeff
    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
    m = cloppa_obj.M(triplet=False)
    assert abs(M_trace - m.trace()) < 1e-5
