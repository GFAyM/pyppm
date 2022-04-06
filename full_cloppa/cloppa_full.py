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



M_list = []
M_diag_list = []
inv_M_diag_list = []

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(ang)
    
    full_M_obj = Cloppa_full(
        mol_input=mol_H2O2,basis='6-31G**',
        mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, mo_occ_loc=mo_occ_loc)
    m = full_M_obj.M
    

    M_list.append([ang*10, np.sum(m)])#,  "Propagador Pol"])
    M_diag_list.append([ang*10, np.sum(np.diag(m))])#,  "Propagador Pol"])
    inv_M_diag_list.append([ang*10, np.sum(np.diag(np.linalg.inv(m)))])#,  "Propagador Pol"])


df = pd.DataFrame(M_list, columns=['angulo', 'Propagator'])#,   'Polarization Propagator'])

df.plot(x='angulo', y='Propagator')

plt.savefig('sum_M.png')

df = pd.DataFrame(M_diag_list, columns=['angulo', 'Propagator'])#,   'Polarization Propagator'])

df.plot(x='angulo', y='Propagator')

plt.savefig('sum_diag_M.png')

df = pd.DataFrame(inv_M_diag_list, columns=['angulo', 'Propagator'])#,   'Polarization Propagator'])

df.plot(x='angulo', y='Propagator')

plt.savefig('sum_inv_diag_M.png')
