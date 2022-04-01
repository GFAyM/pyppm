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



mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(
                                    molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff


full_M_obj = full_M_two_elec(
     mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, mo_occ_loc=mo_occ_loc, 
     occ=[8,7], vir=[13,20,14,21])

b_1 = full_M_obj.M

print(b_1)

cloppa_obj = Cloppa_test(mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, o1=[8], o2=[7], v1=[13,20], v2=[14,21])
m_1 = cloppa_obj.inverse_prop_pol

print(m_1)

print(np.isclose(b_1,m_1))

