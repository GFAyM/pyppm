import pytest
from pyscf import scf
from pyppm.help_functions import extra_functions
from pyppm.entropy import entropy
import os

main_directory=os.path.realpath(os.path.dirname(__file__))+'/../'

@pytest.fixture(scope="module")
def hf_data():
    """
    Fixture that returns the molecule and RHF wavefunction for the HF molecule.
    The SCF result is saved to a checkpoint file.
    """
    molden = main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mf = scf.RHF(mol)
    mf.chkfile = 'HF_test_loc.chk'
    mf.kernel()
    return mol, mf.chkfile

@pytest.mark.parametrize("ent_ab, elec_corr",[(0.11366906583351502, "RPA")])
def test_entropy_ab(ent_ab, elec_corr, hf_data):
    """testing entropy_ab function

    Args:
        ent_ab (real): ent_ab value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, chkfile = hf_data
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    ent_obj = entropy(occ1=[2], occ2=[4], vir1=[8], vir2=[9], 
                      mo_coeff_loc=mo_coeff, mol=mol, elec_corr=str(elec_corr),
                     chkfile=chkfile, z_allexc=True)
    ent_ab_ = ent_obj.entropy_ab
    assert abs(ent_ab - ent_ab_) < 1e-5 

@pytest.mark.parametrize("ent_iaia, elec_corr",[(0.04881348894748024, "RPA")])
def test_entropy_iaia(ent_iaia, elec_corr, hf_data):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, chkfile = hf_data
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    ent_obj = entropy(occ1=[2], occ2=[4], vir1=[8], vir2=[9], 
                      mo_coeff_loc=mo_coeff, mol=mol, elec_corr=str(elec_corr),
                     chkfile=chkfile, z_allexc=True)
    ent_iaia_ = ent_obj.entropy_iaia
    assert abs(ent_iaia - ent_iaia_) < 1e-5

@pytest.mark.parametrize("ent_jbjb, elec_corr",[(0.06485557688599691, "RPA")])
def test_entropy_jbjb(ent_jbjb, elec_corr, hf_data):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, chkfile = hf_data
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    ent_obj = entropy(occ1=[2], occ2=[4], vir1=[8], vir2=[9], 
                      mo_coeff_loc=mo_coeff, mol=mol, elec_corr=str(elec_corr),
                      chkfile=chkfile, z_allexc=True)
    ent_jbjb_ = ent_obj.entropy_jbjb
    assert abs(ent_jbjb - ent_jbjb_) < 1e-5

@pytest.mark.parametrize("ent_ab, elec_corr",[(0.1181616860344455, "HRPA")])
def test_entropy_ab(ent_ab, elec_corr, hf_data):
    """testing entropy_ab function

    Args:
        ent_ab (real): ent_ab value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, chkfile = hf_data
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    ent_obj = entropy(occ1=[2], occ2=[4], vir1=[8], vir2=[9], 
                      mo_coeff_loc=mo_coeff, mol=mol, elec_corr=str(elec_corr),
                     chkfile=chkfile, z_allexc=True)
    ent_ab_ = ent_obj.entropy_ab
    assert abs(ent_ab - ent_ab_) < 1e-5 

@pytest.mark.parametrize("ent_iaia, elec_corr",[(0.0533334556587206, "HRPA")])
def test_entropy_iaia(ent_iaia, elec_corr, hf_data):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, chkfile = hf_data
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    ent_obj = entropy(occ1=[2], occ2=[4], vir1=[8], vir2=[9], 
                      mo_coeff_loc=mo_coeff, mol=mol, elec_corr=str(elec_corr),
                     chkfile=chkfile, z_allexc=True)
    ent_iaia_ = ent_obj.entropy_iaia
    assert abs(ent_iaia - ent_iaia_) < 1e-5

@pytest.mark.parametrize("ent_jbjb, elec_corr",[(0.06482823037568772, "HRPA")])
def test_entropy_jbjb(ent_jbjb, elec_corr, hf_data):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden= main_directory + "tests/HF_cc-pvdz_loc.molden"
    mol, chkfile = hf_data
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    ent_obj = entropy(occ1=[2], occ2=[4], vir1=[8], vir2=[9], 
                      mo_coeff_loc=mo_coeff, mol=mol, elec_corr=str(elec_corr),
                      chkfile=chkfile, z_allexc=True)
    ent_jbjb_ = ent_obj.entropy_jbjb
    assert abs(ent_jbjb - ent_jbjb_) < 1e-5