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
from pyscf import gto, scf, ao2mo


ang=100
mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
'''.format(ang)
mol = gto.M(atom=str(mol_H2O2), basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()

mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang}.molden").extraer_coeff

fock_canonical = mf.get_fock()
#Fock_matrix_loc = mo_coeff_loc.T @ mf.get_fock() @ mo_coeff_loc

#print(Fock_matrix_loc)
occidx = np.where(mo_occ_loc==2)[0]
viridx = np.where(mo_occ_loc==0)[0]

orbo = mo_coeff_loc[:,occidx]
orbv = mo_coeff_loc[:,viridx]
nocc = orbo.shape[1]
nvir = orbv.shape[1]

mo_coeff_loc_occ = mo_coeff_loc[:,:nocc]
mo_coeff_loc_vir = mo_coeff_loc[:,nocc:]


fock_loc_occ = mo_coeff_loc_occ.T @ fock_canonical @ mo_coeff_loc_occ
fock_loc_vir = mo_coeff_loc_vir.T @ fock_canonical @ mo_coeff_loc_vir

#F = np.array([[fock_loc_occ],[fock_loc_vir]],[[fock_loc_occ],[fock_loc_vir]])
