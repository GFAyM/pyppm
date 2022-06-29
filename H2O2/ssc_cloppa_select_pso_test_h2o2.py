import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_100.molden").extraer_coeff
full_M_obj = Cloppa(
    mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
    mo_occ_loc=mo_occ_loc)

#full_M_obj.kernel(FC=False, FCSD=False, PSO=True)
obj = full_M_obj
n_atom1=[2]
n_atom2=[3]

#j = full_M_obj.kernel_pathway_occ(n_atom1=n_atom1, occ_atom1=5, n_atom2=n_atom2, occ_atom2=3)
ssc_tot = 0
m = obj.M(triplet=False)
p = np.linalg.inv(m)
print('i  j   ssc_pso ssc_pso_total')

for i in range(9,38,1):
	for j in range(9,38,1):
#		for a in range(9,38,1):
#			for b in range(9,38,1):
		ssc = full_M_obj.kernel_pathway(FC=False, FCSD=False, PSO=True,
										princ_prop=p,
										n_atom1=n_atom1, occ_atom1=4 , vir_atom1=i, 
										n_atom2=n_atom2, occ_atom2=4, vir_atom2=j)
		ssc_tot += ssc
		if abs(ssc) > 0.1: 
			print(i, j, ssc, ssc_tot)    
print(ssc_tot)
					




