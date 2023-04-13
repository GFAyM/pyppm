import pytest
from src.help_functions import extra_functions
from pyscf import tools

def test_extraer_coeff():
    """testing extraer_coeff property
    """
    molden="C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    mol_t, mo_energy, mo_coeff_t, mo_occ_t, irrep_labels, spins =  tools.molden.load(molden)
    print(mo_occ)
    print(mo_occ_t)
    assert mol.atom_coords().all() == mol_t.atom_coords().all()
    assert mo_coeff.all() == mo_coeff_t.all()
    assert mo_occ.all() == mo_occ_t.all()
    
@pytest.mark.parametrize("atm_id, hibridization_coeff", 
                         [(36, 0.53788694)])

def test_mo_hibridization(atm_id, hibridization_coeff):
    """testing mo_hibridization function

    Args:
        atm_id (int): Atom ID of the atom with the requiered hibridization
        hibridization_coeff (real): hibridization coeff
    """
    extra_func_obj = extra_functions(molden_file="C2H6_ccpvdz_Pipek_Mezey.molden")

    hibri = extra_func_obj.mo_hibridization(lim1=0.5,lim2=0.9, mo_label='H3')
    assert hibri[0] == atm_id
    assert hibri[1] - hibridization_coeff < 1e-8    

@pytest.mark.parametrize("atm_id, hibridization_coeff", 
                         [(36, 0.45948969)])

def test_mo_hibridization_fixed(atm_id, hibridization_coeff):
    """testing mo_hibridization_fixed function

    Args:
        atm_id (int): Atom ID of the atom with the requiered hibridization
        hibridization_coeff (real): hibridization coeff
    """
    extra_func_obj = extra_functions(molden_file="C2H6_ccpvdz_Pipek_Mezey.molden")

    hibri = extra_func_obj.mo_hibridization_fixed(
            fixed_orbital=atm_id,lim1=0.1,lim2=0.9, mo_label='C2')
    assert hibri - hibridization_coeff < 1e-8    


