from pyPPE.help_functions import extra_functions
from pyPPE.entropy import entropy

mol, mo_coeff, mo_occ = extra_functions(molden_file="C2H6_ccpvdz_Pipek_Mezey.molden").extraer_coeff

cloppa_obj = entropy(mo_coeff=mo_coeff, mol=mol,
                     occ=[2,3], vir=[10,11,14,17])

ent_ab = cloppa_obj.entropy_jbjb

print(ent_ab)
