import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.help_functions import extra_functions
from src.cloppa import full_M_two_elec
import plotly.express as px
import pandas as pd
import numpy as np
from pyscf import ao2mo


ang=10



mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff

o1 = mo_coeff_loc[:,[1]]
o2 = mo_coeff_loc[:,[2]]
v1 = mo_coeff_loc[:,[10]]
v2 = mo_coeff_loc[:,[11]]
#

occ = mo_coeff_loc[:,[1,2]]
vir = mo_coeff_loc[:,[10,11]]

nocc = occ.shape[1] 
nvic = vir.shape[1]
mo = np.hstack((occ,vir))

nmo = nocc + nvic

#a_0 = np.zeros((occ.shape[1],vir.shape[1],occ.shape[1],vir.shape[1]))

#eri_mo = ao2mo.general(mol_loc, [occ,mo,mo,mo], compact=False)
#eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)


#eri_reduce = np.einsum('iajb->ijab', eri_mo[:nocc,:nocc,nocc:,nocc:])
#print( eri_mo, eri_mo.shape, a_0.shape)
#print(eri_reduce, eri_reduce.shape)

#ndarray with shape (2,2,2,2)

a = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).reshape((2,2,2,2))

print(a.shape)

#permutation

b = np.einsum('ijab->ijba', a)
print(b)

