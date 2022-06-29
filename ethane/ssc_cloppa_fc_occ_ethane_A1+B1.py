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

if os.path.exists('cloppa_fc_a1+b1_occ_C2H6_ccpvdz.txt'):
	os.remove('cloppa_fc_a1+b1_occ_C2H6_ccpvdz.txt')

H3_1s = [6, 3, 6, 2, 6, 6, 4, 3, 4, 4, 2, 4, 2, 7, 2, 2, 4, 4, 2]
H7_1s = [3, 5, 4, 4, 5, 5, 3, 2, 5, 5, 3, 5, 7, 2, 7, 5, 7, 7, 6]

occ_lmo = [(H3_1s, 'H3_1s'), (H7_1s, 'H7_1s')]

for ang in range(0,18,1):

	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
	ssc_tot = 0
	cloppa_obj = Cloppa(
		mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
		mo_occ_loc=mo_occ_loc)
	
	m = cloppa_obj.M(triplet=True, energy_m=False)
	p = np.linalg.inv(m)

	fc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False, n_atom1=[2], n_atom2=[6], princ_prop=p)

	for i, ii in occ_lmo:
		for j, jj in occ_lmo:
			ssc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
											princ_prop=p,
											n_atom1=[2], occ_atom1=i[ang], 
											n_atom2=[6], occ_atom2=j[ang])
			ssc_tot += ssc
			with open('cloppa_fc_a1+b1_occ_C2H6_ccpvdz.txt', 'a') as f:
				f.write(f'{ang*10} {ssc[0]} {ii} {jj} {fc[0]} \n')        		
	print(ssc_tot, '-----> La suma de las contribuciones para el ángulo', ang*10)
	print(fc, '---------> lo que debería dar la suma de las contribuciones para el ángulo',ang*10)




