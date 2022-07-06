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

H4_1s_occ = [4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 3, 7, 4, 4, 4]
H3_1s_occ = [5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5]

O_O = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8]

occ_lmo = [(O1_1s, 'O1_1s'), (O2_1s, 'O2_1s'), (O1_2p1, 'O1_2p1'), (O2_2p1, 'O2_2p1'), 
		   (O1_2p2, 'O1_2p2'), (O2_2p2, 'O2_2p2'), (H4_1s_occ, 'H4_1s'), (H3_1s_occ, 'H3_1s'), (O_O, 'C_C')]

H3_1s =  [36, 13, 36, 13, 36, 36, 13, 36, 36, 36, 36, 36, 36, 36, 36, 36, 14, 36]
H3_2s =  [12, 11, 12, 11, 12, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 12]
H3_2px =  [29, 22, 29, 23, 29, 28, 29, 28, 27, 27, 27, 28, 30, 29, 29, 29, 23, 29]
H3_2py =  [20, 20, 23, 20, 23, 23, 24, 24, 24, 23, 24, 24, 23, 23, 23, 23, 20, 23]
H3_2pz =  [17, 19, 17, 19, 17, 17, 19, 17, 17, 17, 17, 17, 17, 17, 17, 17, 19, 17]

H4_1s =  [13, 36, 13, 36, 13, 13, 36, 13, 14, 14, 14, 14, 14, 14, 14, 14, 36, 14]
H4_2s =  [11, 12, 11, 12, 11, 11, 12, 11, 13, 13, 13, 13, 13, 13, 13, 13, 12, 13]
H4_2px =  [23, 29, 22, 29, 22, 22, 23, 29, 30, 30, 29, 29, 22, 22, 22, 22, 28, 22]
H4_2py =  [22, 23, 20, 21, 20, 20, 21, 23, 23, 24, 23, 23, 20, 20, 20, 20, 22, 20]
H4_2pz =  [19, 17, 19, 18, 19, 19, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 19]

O1_3dz = [35, 30, 34, 30, 30, 30, 30, 30, 29, 29, 30, 30, 31, 30, 30, 33, 30, 35]
O2_3dz = [30, 35, 30, 35, 31, 31, 35, 31, 32, 32, 31, 31, 27, 31, 31, 30, 35, 30]

O1_3s = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
O2_3s = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

O2_3dx = [16, 14, 16, 14, 16, 16, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16, 11, 16]
O1_3dx = [14, 16, 14, 16, 14, 14, 16, 14, 26, 26, 26, 26, 11, 11, 11, 11, 16, 11]

O2_3py = [24, 24, 18, 24, 18, 18, 20, 18, 18, 18, 18, 18, 18, 18, 18, 18, 24, 18]
O1_3py = [18, 18, 24, 17, 26, 29, 15, 27, 31, 31, 28, 27, 29, 28, 25, 24, 17, 24]
O2_3dy = [32, 31, 31, 31, 32, 32, 31, 33, 34, 35, 33, 33, 33, 33, 33, 31, 31, 31]
O1_3dy = [31, 33, 35, 34, 35, 35, 34, 35, 35, 34, 35, 35, 35, 35, 35, 35, 34, 33]
O2_3dxz = [21, 32, 21, 33, 21, 21, 22, 21, 22, 22, 22, 21, 21, 21, 21, 21, 32, 21]
O1_3dxz = [33, 21, 32, 22, 33, 33, 32, 32, 33, 33, 32, 32, 32, 32, 32, 32, 21, 32]

vir_lmo = [(H3_1s, 'H3_1s'), (H4_1s, 'H4_1s'), (H3_2s, 'H3_2s'), (H4_2s, 'H4_2s'), 
			  (H3_2px, 'H3_2px'), (H4_2px, 'H4_2px'), (H3_2py, 'H3_2py'), (H4_2py, 'H4_2py'), 
				(H3_2pz, 'H3_2pz'),(H4_2pz, 'H4_2pz'), 
			    (O1_3dz, 'O1_3dz'),(O2_3dz, 'O2_3dz'),
				(O1_3s, 'O1_3s'), (O2_3s, 'O2_3s'), (O1_3dx,'O1_3dx'), (O2_3dx, 'O2_3dx'),
				(O1_3py, 'O1_3py'), (O2_3py, 'O2_3py'),(O1_3dy,'O1_3dy'),(O2_3dy,'O2_3dy'),#]#,
				(O1_3dxz,'O1_3ddxz'),(O2_3dxz,'O2_3ddxz')]

for ang in range(1,2,1):
	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
	ssc_tot = 0
	cloppa_obj = Cloppa(
		mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
		mo_occ_loc=mo_occ_loc)
	
	m = cloppa_obj.M(triplet=True)
	p = np.linalg.inv(m)

	fc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False, n_atom1=[2], n_atom2=[3], princ_prop=p)

	for i, ii in occ_lmo:
		for j, jj in occ_lmo:
			for a, aa in vir_lmo:
				for b,bb in vir_lmo:
					ssc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
													princ_prop=p,
													n_atom1=[2], occ_atom1=i[ang-1], 
													n_atom2=[3], occ_atom2=j[ang-1],
													vir_atom1=a[ang], vir_atom2=b[ang])
#					ssc_tot = ssc
					with open('cloppa_fc_iajb_H2O2_631G.txt', 'a') as f:
						f.write(f'{ang*10} {ssc[0]} {ii} {aa} {jj} {bb} {fc[0]} \n')        		
	print(ssc_tot, '-----> La suma de las contribuciones para el ángulo', ang*10)
	print(fc, '---------> lo que debería dar la suma de las contribuciones para el ángulo',ang*10)




