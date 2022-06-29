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

O1_1s = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
O2_1s = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

O1_2p1 = [3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 7, 3, 3, 3, 3]
O2_2p1 = [2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2]

O1_2p2 = [7, 6, 6, 7, 7, 6, 6, 6, 6, 7, 6, 7, 8, 4, 6, 7, 6]
O2_2p2 = [6, 7, 7, 6, 6, 7, 7, 7, 7, 6, 7, 6, 6, 6, 7, 6, 7]

H4_1s = [4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 3, 7, 4, 4, 4]
H3_1s = [5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5]

O_O = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8]

occ_lmo = [(O1_1s, 'O1_1s'), (O2_1s, 'O2_1s'), (O1_2p1, 'O1_2p1'), (O2_2p1, 'O2_2p1'), 
		   (O1_2p2, 'O1_2p2'), (O2_2p2, 'O2_2p2'), (H4_1s, 'H4_1s'), (H3_1s, 'H3_1s'), (O_O, 'C_C')]

for ang in range(1,18,1):

	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
	ssc_tot = 0
	cloppa_obj = Cloppa(
		mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
		mo_occ_loc=mo_occ_loc)
	
	m = cloppa_obj.M(triplet=False)
	p = np.linalg.inv(m)

	pso = cloppa_obj.kernel_pathway(FC=False, FCSD=False, PSO=True, n_atom1=[2], n_atom2=[3],princ_prop=p)

	for i, ii in occ_lmo:
		for j, jj in occ_lmo:
			ssc = cloppa_obj.kernel_pathway(FC=False, FCSD=False, PSO=True,
											princ_prop=p,
											n_atom1=[2], occ_atom1=i[ang-1], 
											n_atom2=[3], occ_atom2=j[ang-1])
			ssc_tot += ssc
			with open('cloppa_pso_occ_H2O2_631G.txt', 'a') as f:
				f.write(f'{ang*10} {ssc[0]} {ii} {jj} {pso[0]} \n')        		
	print(ssc_tot, '-----> La suma de las contribuciones para el ángulo', ang*10)
	print(pso, '---------> lo que debería dar la suma de las contribuciones para el ángulo',ang*10)




