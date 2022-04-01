import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.help_functions import extra_functions
from src.cloppa import full_M_two_elec
from src.cloppa import Cloppa_test

import plotly.express as px
import pandas as pd
import numpy as np
from pyscf import ao2mo


ang=10



mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff


full_M_obj = full_M_two_elec(mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, mo_occ_loc=mo_occ_loc)#, occ=[1,2], vir=[10,11])

m = full_M_obj.M
print(m.shape)


print(m[0][0])






# (ab|ji) - (aj|bi)
#int_test = -ao2mo.general(mol_loc,[v1,v1,o1,o1]) - ao2mo.general(mol_loc, [v1,o1,v1,o1])
#print(int_test)

#int_test = -ao2mo.general(mol_loc,[v1,v2,o1,o1]) - ao2mo.general(mol_loc, [v1,o1,v2,o1])
#print(int_test)

#int_test = -ao2mo.general(mol_loc,[v1,v1,o2,o1]) - ao2mo.general(mol_loc, [v1,o2,v1,o1])
#print(int_test)

