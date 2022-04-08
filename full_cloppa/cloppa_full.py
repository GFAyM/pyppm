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


M_list = [[],[]]
M_diag_list = [[],[]]
inv_M_diag_list = [[],[]]
inv_M_list = [[],[]]

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(10*ang)
    
    full_M_obj = Cloppa_full(
        mol_input=mol_H2O2,basis='6-31G**',
        mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, mo_occ_loc=mo_occ_loc)
    
    m = full_M_obj.M
    

    M_list[0].append(ang*10)#, np.sum(m)])#,  "Propagador Pol"])
    M_list[1].append(np.sum(m))
    M_diag_list[0].append(ang*10)#,  "Propagador Pol"])
    M_diag_list[1].append(np.sum(np.diag(m)))#,  "Propagador Pol"])
    inv_M_diag_list[0].append(ang*10)
    inv_M_diag_list[1].append(np.sum(np.diag(np.linalg.inv(m))))#,  "Propagador Pol"])
    inv_M_list[0].append(ang*10)
    inv_M_list[1].append(np.sum(np.linalg.inv(m)))#,  "Propagador Pol"])


fig = plt.figure(figsize=(8, 8))
x = M_list[0]
y = M_list[1]
plt.plot(x,y)
plt.title("sum of elements of M matrix in a Localized MO basis")
plt.savefig('sum_localized_M.png')

fig = plt.figure(figsize=(12, 8))
y = M_diag_list[1]
plt.plot(x,y)
plt.title("sum of elements of the diagonal of M matrix in Localized MO basis")
plt.savefig('sum_diag_Localized_M.png')

fig = plt.figure(figsize=(8, 8))
y = inv_M_diag_list[1]
plt.plot(x,y)
plt.title("sum of elements of diagonal of the Principal Propagator matrix in the Localized MO basis")
plt.savefig('sum_inv_diag_Localized_M.png')

fig = plt.figure(figsize=(8, 8))
y = inv_M_list[1]
plt.plot(x,y)
plt.title("sum of elements of Principal Propagator matrix in the Localized MO basis")
plt.savefig('sum_inv_Localized_M.png')


