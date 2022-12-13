import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)


from src.help_functions import extra_functions

from src.cloppa import Cloppa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

text = str('elements_H2O2.txt')
if os.path.exists(text):
	os.remove(text)



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

H4_1s_occ = [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 3, 7, 4, 4, 4]
H3_1s_occ = [5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5]

for ang in range(0,18,1):
	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff

	occ = [H3_1s_occ[ang],H4_1s_occ[ang]]
	vir = [H3_1s[ang],H3_2s[ang],H3_2px[ang],H3_2pz[ang],
		   H4_1s[ang],H4_2s[ang],H4_2px[ang],H4_2pz[ang]]
	cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc,	mo_occ_loc=mo_occ_loc)

	m = cloppa_obj.M(triplet=True)

	p = np.linalg.inv(m)

	p1, m, p2 = cloppa_obj.kernel_pathway_elements(FC=True, FCSD=False, PSO=False,
							princ_prop=p,
							n_atom1=[2], occ_atom1=H3_1s_occ[ang], vir_atom1=H3_2s[ang],
							n_atom2=[3], occ_atom2=H4_1s_occ[ang], vir_atom2=H4_2s[ang], elements=True)

	with open(text, 'a') as f:
		f.write(f'{ang*10} {p1} {p2} {m}\n')			

df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang','p1','p2','m']

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))
#plt.figure(figsize=(10,8))
ax1.plot(df.ang, df.p1, 'b>-', label='H1') #f'a={orb1} b={orb2}')
ax1.set_title(r'${b}^{FC}_{i=OH,a=OH1**}$')
ax1.legend()
ax3.plot(df.ang, df.p2, 'b>-', label='H2' )#f'a={orb1} b={orb2}')
ax3.set_title(r'${b}^{FC}_{j=OH2,b=OH2**}$')
ax3.legend()
ax2.plot(df.ang, df.m, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax2.set_title(r'$^3{P}_{ia,jb}$')
#plt.ylabel('Hz')
ax1.set_xlabel('Ángulo diedro')
ax2.set_xlabel('Ángulo diedro')
ax3.set_xlabel('Ángulo diedro')

ax4.plot(df.ang, df.p1*df.p2*df.m, 'r>-')
ax4.set_title(r'${b}^{FC}_{H_1,ia}$* $^3{P}_{ia,jb}$* ${b}^{FC}_{H_2,jb}$')
ax4.yaxis.tick_right()
ax4.set_xlabel('Angulo diedro')
plt.suptitle(r'''Elements of Polarization Propagator $J^{FC}(H-H)$ in H$_2$O$_2$''')
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
plt.savefig('FC_elements_H2O2_OH1_OH2.png', dpi=200)
plt.show()  