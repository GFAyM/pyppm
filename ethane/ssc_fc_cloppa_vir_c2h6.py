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

if os.path.exists('cloppa_fc_iajb_C2H6_ccpvdz.txt'):
    os.remove('cloppa_fc_iajb_C2H6_ccpvdz.txt')


H3_2s = [19, 19, 19, 20, 21, 19, 22, 19, 19, 19, 19, 19, 19, 19, 24, 20, 22, 21, 22]
H7_2s = [23, 23, 24, 22, 23, 21, 23, 20, 20, 20, 21, 21, 24, 20, 21, 24, 23, 24, 19]

H3_2px = [44, 52, 52, 52, 54, 40, 44, 40, 52, 52, 52, 48, 52, 40, 53, 52, 52, 40, 44]
H7_2px =[50, 45, 37, 48, 50, 48, 45, 51, 54, 47, 50, 47, 54, 47, 45, 47, 47, 54, 47]

H3_2py = [51, 51, 51, 51, 53, 46, 51, 46, 51, 51, 51, 50, 51, 48, 48, 51, 51, 46, 51]
H3_2pz = [17, 17, 17, 17, 17, 17, 17, 18, 17, 17, 17, 17, 18, 17, 17, 17, 17, 17, 17]

#         0    1   2  3    4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
H7_2py = [49, 49, 43, 45, 43, 44, 40, 47, 53, 50, 47, 45, 53, 41, 42, 44, 46, 49, 48]
H7_2pz = [18, 18, 39, 40, 42, 41, 43, 56, 39, 18, 18, 40, 37, 45, 38, 41, 18, 18, 18]

H3_1s = [25, 36, 53, 36, 55, 31, 37, 31, 35, 36, 53, 27, 36, 35, 30, 36, 35, 31, 37]
H7_1s = [54, 37, 26, 28, 25, 28, 25, 57, 55, 35, 36, 25, 55, 28, 56, 25, 25, 33, 36]

H3_1s_occ = [6, 3, 6, 2, 6, 6, 4, 3, 4, 4, 2, 4, 2, 7, 2, 2, 4, 4, 2]
H7_1s_occ = [3, 5, 4, 4, 5, 5, 3, 2, 5, 5, 3, 5, 7, 2, 7, 5, 7, 7, 6]

occ_lmo = [(H3_1s_occ, 'lig_1'), (H7_1s_occ, 'lig_2')]

vir_lmo = [(H3_1s, 'H3_1s'), (H7_1s, 'H7_1s'), (H3_2s, 'H3_2s'), (H7_2s, 'H7_2s'), 
			  (H3_2px, 'H3_2px'), (H7_2px, 'H7_2px'), (H3_2py, 'H3_2py'), (H7_2py, 'H7_2py'), 
				(H3_2pz, 'H3_2pz'),(H7_2pz, 'H7_2pz')]

for ang in range(0,19,1):

	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
	ssc_tot = 0
	cloppa_obj = Cloppa(
		mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
		mo_occ_loc=mo_occ_loc)
	
	m = cloppa_obj.M(triplet=True)
	p = np.linalg.inv(m)

	fc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False, n_atom1=[2], n_atom2=[6],princ_prop=p,
									occ_atom1=H3_1s_occ[ang], occ_atom2=H7_1s_occ[ang])

	for i, ii in occ_lmo:
		for j, jj in occ_lmo:
			for a, aa in vir_lmo:
				for b, bb in vir_lmo:
					fc_iajb = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
													princ_prop=p,
													n_atom1=[2],n_atom2=[6],
													occ_atom1=i[ang], 
													occ_atom2=j[ang],
													vir_atom1=a[ang], vir_atom2=b[ang])
					#if abs(ssc) > 0.001 :
					#ssc_tot += ssc
					with open('cloppa_fc_iajb_C2H6_ccpvdz.txt', 'a') as f:
						f.write(f'{ang*10} {np.round(fc_iajb[0], decimals=6)} {ii} {aa} {jj} {bb} {np.round(fc[0],decimals=6)} \n')        		

