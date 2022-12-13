from cProfile import label
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
from pyscf import lib

#print('number of threads:',lib.num_threads())
text = str('elementos_cloppa_C2H6_fc_iajb.txt')
if os.path.exists(text):
	os.remove(text)

H3_2s = [19, 19, 19, 20, 21, 19, 22, 19, 19, 19, 19, 19, 19, 19, 24, 20, 22, 21, 22]
H7_2s = [23, 23, 24, 22, 23, 21, 23, 20, 20, 20, 21, 21, 24, 20, 21, 24, 23, 24, 19]

H3_2px = [44, 52, 52, 52, 54, 40, 44, 40, 52, 52, 52, 48, 52, 40, 53, 52, 52, 40, 44]
H7_2px = [50, 45, 37, 48, 50, 48, 45, 51, 54, 47, 50, 47, 54, 47, 45, 47, 47, 54, 47]

H3_2py = [51, 51, 51, 51, 53, 46, 51, 46, 51, 51, 51, 50, 51, 48, 48, 51, 51, 46, 51]
H3_2pz = [17, 17, 17, 17, 17, 17, 17, 18, 17, 17, 17, 17, 18, 17, 17, 17, 17, 17, 17]

#         0    1   2  3    4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
H7_2py = [49, 49, 43, 45, 43, 44, 43, 47, 53, 50, 47, 45, 53, 41, 42, 44, 46, 49, 48]
H7_2pz = [18, 18, 39, 40, 42, 41, 40, 56, 39, 18, 18, 40, 37, 45, 38, 41, 18, 18, 18]

H3_1s = [25, 36, 53, 36, 55, 31, 37, 31, 35, 36, 53, 27, 36, 35, 30, 36, 35, 31, 37]
H7_1s = [54, 37, 26, 28, 25, 28, 25, 57, 55, 35, 36, 25, 55, 28, 56, 25, 25, 33, 36]

H3_1s_occ = [6, 3, 6, 2, 6, 6, 4, 3, 4, 4, 2, 4, 2, 7, 2, 2, 4, 4, 2]
H7_1s_occ = [3, 5, 4, 4, 5, 5, 3, 2, 5, 5, 3, 5, 7, 2, 7, 5, 7, 7, 6]

#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27




for ang in range(0,18,1):
	mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
	cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, #vir=viridx, occ=occidx,
	mo_occ_loc=mo_occ)

	m = cloppa_obj.M(triplet=True)

	p = np.linalg.inv(m)

	p1, m, p2 = cloppa_obj.kernel_pathway_elements(FC=True, FCSD=False, PSO=False,
							princ_prop=p,
							n_atom1=[2], occ_atom1=H3_1s_occ[ang], vir_atom1=H3_1s[ang],
							n_atom2=[6], occ_atom2=H7_1s_occ[ang], vir_atom2=H7_1s[ang], elements=True)

	with open(text, 'a') as f:
		f.write(f'{ang*10} {abs(p1)} {abs(p2)} {abs(m)}\n')

#					print(p1, m, p2)

df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang','p1','p2','m']

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))
#plt.figure(figsize=(10,8))
ax1.plot(df.ang, df.p1, 'b>-', label='H1') #f'a={orb1} b={orb2}')
ax1.set_title(r'${b}^{FC}_{i=CH1,a=CH1**}$')
ax1.legend()
ax3.plot(df.ang, df.p2, 'b>-', label='H2' )#f'a={orb1} b={orb2}')
ax3.set_title(r'${b}^{FC}_{j=CH1,b=CH1**}$')
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
plt.suptitle(r'''Elements of Polarization Propagator $J^{FC}(H-H)$ in C$_2$H$_6$''')
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
plt.savefig('FC_elements_C2H6_CH1_CH1**_CH1_CH1**.png', dpi=200)
plt.show()  