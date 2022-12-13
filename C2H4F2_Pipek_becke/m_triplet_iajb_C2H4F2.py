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

#print('number of threads:',lib.num_threads())
text = str('elementos_cloppa_C2H4F2_fcsd_iajb.txt')
if os.path.exists(text):
	os.remove(text)

lig1 = [9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
lig2 = [8, 8, 8, 8, 8, 7, 8, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8]

par_lib_1 = [10, 11, 10, 10, 10, 11, 11, 11, 10, 11, 11, 10, 11, 10, 11, 11, 11, 10, 11]
par_lib_2 = [11, 10, 11, 11, 11, 10, 10, 10, 11, 10, 10, 11, 10, 11, 10, 10, 10, 11, 10]

#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
#occ_lmo = [(occ1,'O-H1'), (occ2,'O-H2')]

v1_1 = [31, 74, 74, 73, 73, 73, 73, 73, 74, 74, 74, 73, 73, 73, 73, 73, 73, 73, 74]
v1_2 = [73, 31, 31, 31, 31, 74, 74, 74, 30, 30, 30, 74, 74, 74, 31, 31, 31, 31, 31]

v2_1 = [29, 50, 39, 40, 40, 40, 39, 40, 32, 31, 32, 33, 33, 40, 40, 40, 40, 41, 40]
v2_2 = [49, 29, 29, 23, 25, 23, 22, 32, 31, 32, 31, 22, 23, 23, 25, 29, 22, 29, 29] #mezcla de 3pz y 3s

v3_1 = [41, 32, 32, 32, 32, 33, 33, 33, 40, 41, 40, 40, 39, 33, 32, 32, 32, 32, 32]
v3_2 = [32, 41, 40, 41, 41, 50, 43, 41, 39, 40, 39, 41, 43, 41, 44, 43, 41, 40, 41] #mezcla de 3s y 3pz
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
v4_1 = [45, 40, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 44, 45] #px
v4_2 = [40, 48, 48, 48, 46, 41, 47, 48, 49, 46, 49, 48, 47, 46, 41, 42, 48, 45, 44]

v5_1 = [50, 45, 51, 51, 51, 51, 51, 50, 50, 50, 50, 49, 49, 51, 50, 51, 50, 51, 50] #py
v5_2 = [46, 51, 50, 49, 50, 44, 50, 49, 51, 49, 51, 50, 51, 49, 48, 49, 49, 50, 49]




for ang in range(0,18,1):
	mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H4F2_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
	cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, #vir=viridx, occ=occidx,
	mo_occ_loc=mo_occ)

	m = cloppa_obj.M(triplet=True, energy_m=True)
	
	nocc = cloppa_obj.nocc
	nvir = cloppa_obj.nvir
	m = m.reshape(nocc,nvir,nocc,nvir)

	
	m_ = m[lig1[ang],v1_1[ang]-nocc,lig2[ang],v1_2[ang]-nocc]
	m_ += m[lig1[ang],v1_1[ang]-nocc,lig2[ang],v2_2[ang]-nocc]
	m_ += m[lig1[ang],v2_1[ang]-nocc,lig2[ang],v2_2[ang]-nocc]
	m_ += m[lig1[ang],v3_1[ang]-nocc,lig2[ang],v1_2[ang]-nocc]
	m_ += m[lig1[ang],v3_1[ang]-nocc,lig2[ang],v2_2[ang]-nocc]
	m_ += m[lig1[ang],v3_1[ang]-nocc,lig2[ang],v3_2[ang]-nocc]

#	print(m_)

	with open(text, 'a') as f:
		f.write(f'{ang*10} {m_}\n')


df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang','m']

plt.figure(figsize=(10,10))
plt.plot(df.ang, df.m, 'b^-', label='M')
plt.legend()
plt.ylabel('Hz')
plt.xlabel('Dihedral Angle')
#plt.title('J-coupling $^3J$(F-F) in C$_2$H$_4$F$_2$ with cc-pVDZ basis')
#plt.savefig('mecanismos_C2H4F2_ccpvdz.png', dpi=400)
plt.show()

