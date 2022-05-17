import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa_full
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import ao2mo


M_list = [[],[]]

for ang in range(10,11,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(10*ang)

    viridx = np.where(mo_occ_loc==0)[0]
    occidx = np.where(mo_occ_loc==2)[0]
    full_M_obj = Cloppa_full(
        mol_input=mol_H2O2,basis='6-31G**',
        mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, vir=viridx,
        mo_occ_loc=mo_occ_loc)

    m = full_M_obj.M
    p = np.linalg.inv(m)
    p_reshaped = p.reshape(9,29,9,29)
    elements = p_reshaped[8,[10,11],7,[20,21]]
    suma = np.sum(elements)
    print(elements)
    print(suma)
#    print(p_reshaped[1,1,1,1])
