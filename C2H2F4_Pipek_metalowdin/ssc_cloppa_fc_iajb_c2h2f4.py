import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.cloppa import Cloppa
import numpy as np

#print('number of threads:',lib.num_threads())
if os.path.exists('cloppa_fc_iajb_C2H4F2.txt'):
	os.remove('cloppa_fc_iajb_C2H4F2.txt')


lig_1 = [22, 22, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 23, 22, 23, 23, 23, 22]
lig_2 = [23, 23, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 22, 23, 22, 22, 22, 23]

occ_lmo = [(lig_1,'O-H1'), (lig_2,'O-H2')]

vir1_1s = [63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63]
vir2_1s = [64, 64, 64, 64, 53, 64, 53, 64, 64, 64, 53, 64, 53, 64, 64, 64, 64, 64]

vir1_2s = [36, 36, 36, 36, 36, 35, 35, 36, 36, 36, 35, 35, 36, 36, 36, 36, 36, 36]
vir2_2s = [35, 35, 35, 35, 35, 34, 34, 35, 35, 35, 34, 34, 35, 35, 35, 35, 35, 35]

vir1_2px = [66, 67, 67, 67, 67, 67, 67, 65, 65, 65, 67, 67, 67, 67, 67, 67, 66, 65]
vir2_2px = [65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 65, 65, 66]

vir1_2py = [68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 67]

vir1_2pz = [38, 38, 38, 39, 39, 39, 39, 39, 38, 39, 39, 39, 39, 39, 38, 38, 38, 37]

vir2_2py = [67, 66, 65, 65, 65, 65, 65, 67, 67, 67, 65, 65, 65, 38, 65, 66, 67, 68]
vir2_2pz = [37, 37, 37, 38, 64, 38, 38, 38, 39, 38, 38, 38, 64, 65, 37, 37, 37, 38]

lmo_vir = [(vir1_1s,"H1_1s"),(vir2_1s,"H2_1s"),(vir1_2s,"H1_2s"),
			(vir2_2s,"H2_2s"), (vir1_2px,"H1_2px"),
		   (vir2_2px,"H2_2px"), (vir1_2py,"H1_2py"),
		   (vir2_2py,"H2_2py"), (vir1_2pz,"H1_2pz"), (vir2_2pz,"H2_2pz")]

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
	for i, ii in occ_lmo:
		for j, jj in occ_lmo:
			for a, aa in lmo_vir:
				for b, bb in lmo_vir:		
					fc_iajb = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
															princ_prop=p,
															n_atom1=[2], occ_atom1=i[ang-10], vir_atom1=a[ang-10], 
															n_atom2=[6], occ_atom2=j[ang-10], vir_atom2=b[ang-10])
					with open('cloppa_fc_iajb_C2H4F2.txt', 'a') as f:
						f.write(f'{ang*10} {fc_iajb[0]} {ii} {aa} {jj} {bb} \n')
	#print(fc_iajb, '-----> La suma de las contribuciones para el ángulo', ang*10)
	#print(fc, '---------> lo que debería dar la suma de las contribuciones para el ángulo',ang*10)
