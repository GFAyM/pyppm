import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.cloppa import Cloppa
import numpy as np

#print('number of threads:',lib.num_threads())
if os.path.exists('cloppa_fc_ij_C2H4F2.txt'):
	os.remove('cloppa_fc_ij_C2H4F2.txt')


lig_1 = [22, 22, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 23, 22, 23, 23, 23, 22]
lig_2 = [23, 23, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 22, 23, 22, 22, 22, 23]



occ_lmo = [(lig_1,'F3_2p2'), (lig_2,'F7_2p2')]



#lmo_vir = [(v1_1,"F3_2pz"),(v1_2,"F7_2pz"),(v2_1,"F3_3pz"),(v2_2,"F7_3pz"), (v3_1,"F3_3s"),(v3_2,"F7_3s"),
#			(v4_1,"F3_3dz"),(v4_2,"F7_3dz"), (v5_1,"F3_3py"), (v5_2,"F7_3py"),(v6_1,"F3_3px"),(v6_2,"F7_3px"),
#			(v7_1,"F3_3dxy"),(v7_2,"F7_3dxy"), (v8_1,"F3_3dx2-y2"),(v8_2,"F7_3dx2-y2"), 
#			(v9_1,"F3_3dyz"),(v9_2,"F7_3dyz"), (v10_1,"F3_3dxz"), (v10_2, "F7_3dxz"), (v_CC, 'V_CC') ]

for ang in range(10,28,1):
	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(
							molden_file=f"C2H2F2_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
	cloppa_obj = Cloppa(
				mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
				mo_occ_loc=mo_occ_loc)
	m = cloppa_obj.M(triplet=True)
	p = np.linalg.inv(m)
	#ssc_tot = 0
	#fcsd=0
#	for i, ii in occ_lmo:
#		for j, jj in occ_lmo:
	fc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
											princ_prop=p, all_pathways=True,
											n_atom1=[2], 
											n_atom2=[6])
#			
	fc_ij = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
											princ_prop=p,
											n_atom1=[2], occ_atom1=[lig_1[ang-10],lig_2[ang-10]], 
											n_atom2=[6], occ_atom2=[lig_1[ang-10],lig_2[ang-10]])
	with open('cloppa_fc_ij_C2H4F2.txt', 'a') as f:
		f.write(f'{ang*10} {fc[0]} {fc_ij[0]} \n')
	print(fc_ij, '-----> La suma de las contribuciones para el ángulo', ang*10)
	print(fc, '---------> lo que debería dar la suma de las contribuciones para el ángulo',ang*10)
