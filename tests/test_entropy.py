import pytest
from src.help_functions import extra_functions
from src.entropy import entropy


@pytest.mark.parametrize("ent_ab", 
                         [(1.3715180507864897)])

def test_entropy_ab(ent_ab):
    """testing entropy_ab function

    Args:
        ent_ab (real): ent_ab value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden="C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff

    cloppa_obj = entropy(mo_coeff=mo_coeff, mol=mol,
                        occ=[2,3], vir=[10,11,14,17])
    ent_ab_ = cloppa_obj.entropy_ab
    assert ent_ab - ent_ab_ < 1e-5

@pytest.mark.parametrize("ent_iaia", 
                         [(0.7108313054501851)])

def test_entropy_iaia(ent_iaia):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden="C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    cloppa_obj = entropy(mo_coeff=mo_coeff, mol=mol,
                        occ=[2,3], vir=[10,11,14,17])
    ent_iaia_ = cloppa_obj.entropy_iaia
    assert ent_iaia - ent_iaia_ < 1e-5

@pytest.mark.parametrize("ent_jbjb", 
                         [(0.6606867886260831)])

def test_entropy_jbjb(ent_jbjb):
    """testing entropy_iaia function

    Args:
        ent_ab (real): ent_iaia value given a couple of occupied LMOs and 
        two pairs of virtual LMOs
    """
    molden="C2H6_ccpvdz_Pipek_Mezey.molden"
    mol, mo_coeff, mo_occ = extra_functions(molden_file=molden).extraer_coeff
    cloppa_obj = entropy(mo_coeff=mo_coeff, mol=mol,
                        occ=[2,3], vir=[10,11,14,17])
    ent_jbjb_ = cloppa_obj.entropy_jbjb
    assert ent_jbjb - ent_jbjb_ < 1e-5
