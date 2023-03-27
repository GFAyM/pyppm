import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.polaritization_propagator import Prop_pol as pp
from src.help_functions import extra_functions
from src.cloppa import Cloppa
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf

file = 'cloppa_pso_ij_C2H4F2.txt'

if os.path.exists(file):
	os.remove(file)

lig1 = [9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
lig2 = [8, 8, 8, 8, 8, 7, 8, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8]

par_lib_1 = [10, 11, 10, 10, 10, 11, 11, 11, 10, 11, 11, 10, 11, 10, 11, 11, 11, 10, 11]
par_lib_2 = [11, 10, 11, 11, 11, 10, 10, 10, 11, 10, 10, 11, 10, 11, 10, 10, 10, 11, 10]

par_libx_1 = [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5]
par_libx_2 = [4, 5, 6, 6, 6, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5, 4]

par_liby_1 = [7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7]
par_liby_2 = [6, 7, 7, 7, 7, 8, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
#occ_lmo = [(occ1,'O-H1'), (occ2,'O-H2')]

v1_1 = [31, 74, 74, 73, 73, 73, 73, 73, 74, 74, 74, 73, 73, 73, 73, 73, 73, 73, 74]
v1_2 = [73, 31, 31, 31, 31, 74, 74, 74, 30, 30, 30, 74, 74, 74, 31, 31, 31, 31, 31]

v2_1 = [29, 50, 39, 40, 40, 40, 39, 40, 32, 31, 32, 33, 33, 40, 40, 40, 40, 41, 40]
v2_2 = [49, 29, 29, 23, 25, 23, 22, 32, 31, 32, 31, 22, 23, 23, 25, 29, 22, 29, 29] #mezcla de 3pz y 3s

v3_1 = [41, 32, 32, 32, 32, 33, 33, 33, 40, 41, 40, 40, 39, 33, 32, 32, 32, 32, 32]
v3_2 = [32, 41, 40, 41, 41, 50, 43, 41, 39, 40, 39, 41, 43, 41, 44, 43, 41, 40, 41] #mezcla de 3s y 3pz
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
v4_1 = [45, 40, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 44, 45]
v4_2 = [40, 48, 48, 48, 46, 41, 47, 48, 49, 46, 49, 48, 47, 46, 41, 42, 48, 45, 44]

v5_1 = [50, 45, 51, 51, 51, 51, 51, 50, 50, 50, 50, 49, 49, 51, 50, 51, 50, 51, 50]
v5_2 = [46, 51, 50, 49, 50, 44, 50, 49, 51, 49, 51, 50, 51, 49, 48, 49, 49, 50, 49]


occ_lmo = [(lig1,'F3_2pz'), (lig2,'F7_2pz'), (par_lib_1,'F3_2s'), (par_lib_2,'F7_2s'),
            (par_libx_1,'F3_LPx'),(par_libx_2,'F7_LPx'),
            (par_liby_1,'F3_LPy'),(par_liby_2,'F7_LPy')]



lmo_vir = [(v1_1,"F3_2pz"),(v1_2,"F7_2pz"),(v2_1,"F3_3pz"),(v2_2,"F7_3pz"), (v3_1,"F3_3s"),(v3_2,"F7_3s"),
			(v4_1,"F3_3py"),(v4_2,"F7_3py"), (v5_1,"F3_3px"), (v5_2,"F7_3px")]

lmo_vir1 = [(v1_1,"F3_2pz_"),(v2_1,"F3_3pz_"),(v3_1,"F3_3s_"),
			(v4_1,"F3_3py_"),(v5_1,"F3_3px_")]

lmo_vir2 = [(v1_2,"F7_2pz_"),(v2_2,"F7_3pz_"), (v3_2,"F7_3s_"),
			(v4_2,"F7_3py_"),(v5_2,"F7_3px_")]



for ang in range(0,18,1):
	ssc_tot = 0
	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(
		molden_file=f"C2H4F2_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff    
	cloppa_obj = Cloppa(
				mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
				mo_occ_loc=mo_occ_loc)
	
	m = cloppa_obj.M(triplet=False)
	p = np.linalg.inv(m)
	for i, ii in occ_lmo:
		for j, jj in occ_lmo:
			ssc = cloppa_obj.kernel_pathway(FC=False, FCSD=False, PSO=True,
											princ_prop=p,
											n_atom1=[2], occ_atom1=i[ang], 
											n_atom2=[6], occ_atom2=j[ang])
			with open(file, 'a') as f:
				f.write(f'{ang*10} {ssc[0]} {ii} {jj} \n')     
				