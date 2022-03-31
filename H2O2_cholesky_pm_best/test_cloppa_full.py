import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append("/home/fer/pyPPE/src")
from help_functions import extra_functions
from cloppa import full_M_two_elec
import plotly.express as px
import pandas as pd
import numpy as np
from pyscf import ao2mo


ang=10



mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff


full_M_obj = full_M_two_elec(mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, mo_occ_loc=mo_occ_loc, occ=[1,2], vir=[10,11])

b_1 = full_M_obj.M

print(b_1)

o1 = mo_coeff_loc[:,[1]]
o2 = mo_coeff_loc[:,[2]]
v1 = mo_coeff_loc[:,[10]]
v2 = mo_coeff_loc[:,[11]]
#




# (ab|ji) - (aj|bi)
int_test = -ao2mo.general(mol_loc,[v1,v1,o1,o1]) - ao2mo.general(mol_loc, [v1,o1,v1,o1])
print(int_test)

int_test = -ao2mo.general(mol_loc,[v1,v2,o1,o1]) - ao2mo.general(mol_loc, [v1,o1,v2,o1])
print(int_test)

int_test = -ao2mo.general(mol_loc,[v1,v1,o2,o1]) - ao2mo.general(mol_loc, [v1,o2,v1,o1])
print(int_test)

