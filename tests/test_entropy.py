import pytest
from pyscf import scf
from pyppm.help_functions import extra_functions
from pyppm.entropy import entropy
import os

main_directory=os.path.realpath(os.path.dirname(__file__))+'/../'

@pytest.mark.parametrize("ent_ab, elec_corr",[(0.6763993986072987, "RPA")])
def test_entropy_ab(ent_ab, elec_corr):
    """testing entropy_ab function

    Args:
        ent_ab (real): ent_ab value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    ent_obj = entropy(occ=[2,4], vir=[8,9], mo_coeff_loc=mo_coeff, mf=mf, 
                      elec_corr = elec_corr, z_allexc=False)
    ent_ab_ = ent_obj.entropy_ab
    assert abs(ent_ab - ent_ab_) < 1e-5 

@pytest.mark.parametrize("ent_iaia, elec_corr",[(0.36568909801148347, "RPA")])
def test_entropy_iaia(ent_iaia, elec_corr):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    #molden="C2H6_ccpvdz_Pipek_Mezey.molden"
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    ent_obj = entropy(occ=[2,4], vir=[8,9], mo_coeff_loc=mo_coeff, mf=mf, 
                      elec_corr=elec_corr, z_allexc=False)
    ent_iaia_ = ent_obj.entropy_iaia
    assert abs(ent_iaia - ent_iaia_) < 1e-5

@pytest.mark.parametrize("ent_jbjb, elec_corr",[(0.3107103005966103, "RPA")])
def test_entropy_jbjb(ent_jbjb, elec_corr):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    #molden="C2H6_ccpvdz_Pipek_Mezey.molden"
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    ent_obj = entropy(occ=[2,4], vir=[8,9], mo_coeff_loc=mo_coeff, mf=mf, 
                      elec_corr=elec_corr, z_allexc=False)
    ent_jbjb_ = ent_obj.entropy_jbjb
    assert abs(ent_jbjb - ent_jbjb_) < 1e-5

@pytest.mark.parametrize("ent_ab, elec_corr",[(0.6851136364894825, "HRPA")])
def test_entropy_ab(ent_ab, elec_corr):
    """testing entropy_ab function

    Args:
        ent_ab (real): ent_ab value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    ent_obj = entropy(occ=[2,4], vir=[8,9], mo_coeff_loc=mo_coeff, mf=mf, 
                      elec_corr=elec_corr, z_allexc=False)
    ent_ab_ = ent_obj.entropy_ab
    assert abs(ent_ab - ent_ab_) < 1e-5

@pytest.mark.parametrize("ent_iaia, elec_corr",[(0.36180860095233963, "HRPA")])
def test_entropy_iaia(ent_iaia, elec_corr):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    #molden="C2H6_ccpvdz_Pipek_Mezey.molden"
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    ent_obj = entropy(occ=[2,4], vir=[8,9], mo_coeff_loc=mo_coeff, mf=mf, 
                      elec_corr=elec_corr, z_allexc=False)
    ent_iaia_ = ent_obj.entropy_iaia
    assert abs(ent_iaia - ent_iaia_) < 1e-5

@pytest.mark.parametrize("ent_jbjb, elec_corr",[(0.32330503553791234, "HRPA")])
def test_entropy_jbjb(ent_jbjb, elec_corr):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    #molden="C2H6_ccpvdz_Pipek_Mezey.molden"
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.kernel()
    ent_obj = entropy(occ=[2,4], vir=[8,9], mo_coeff_loc=mo_coeff, mf=mf, 
                      elec_corr=elec_corr, z_allexc=False)
    ent_jbjb_ = ent_obj.entropy_jbjb
    assert abs(ent_jbjb - ent_jbjb_) < 1e-5