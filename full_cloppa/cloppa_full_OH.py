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

for ang in range(1,18,1):
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

    viridx_OH2_1s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H4', .5, .7)

    occidx_OH2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H4', .3, .5)

    viridx_OH1_1s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H3', .5, .7)

    occidx_OH1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H3', .3, .5)

    #m = full_M_obj.M
    oh_matrix_elelment = full_M_obj.elements_p(occidx_OH1,viridx_OH1_1s,occidx_OH2,viridx_OH2_1s)

    M_list[0].append(ang*10)#, np.sum(m)])#,  "Propagador Pol"])
    M_list[1].append(abs(oh_matrix_elelment))



fig = plt.figure(figsize=(8, 8))
x = M_list[0]
y = M_list[1]
plt.plot(x,y)
plt.title("sum of elements of M matrix for O-H")
#plt.savefig('sum_localized_M.png')
plt.show()


