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

H3_1s_occ = [3, 3, 7, 5, 3, 2, 5, 2, 5, 2, 3, 3, 3, 3, 3, 2, 3, 3, 4]
H7_1s_occ = [5, 6, 4, 6, 6, 7, 6, 3, 4, 3, 2, 2, 7, 7, 4, 6, 7, 6, 6]

H3_1s = [17, 17, 17, 42, 24, 24, 26, 49, 24, 20, 17, 27, 17, 27, 29, 17, 24, 27, 24]
H7_1s = [18, 18, 18, 18, 29, 31, 49, 18, 26, 27, 26, 33, 55, 38, 55, 18, 18, 28, 18]

H3_2pz = [29, 37, 29, 36, 30, 37, 39, 56, 30, 36, 29, 45, 53, 45, 54, 55, 30, 56, 39]
H7_2pz = [34, 35, 35, 40, 51, 52, 56, 39, 37, 56, 37, 52, 38, 56, 45, 38, 38, 55, 37]

H3_2s = [16, 21, 24, 22, 16, 16, 24, 20, 16, 16, 24, 20, 23, 20, 23, 24, 16, 19, 16]
H7_2s = [21, 22, 19, 20, 21, 20, 22, 19, 23, 19, 23, 22, 22, 19, 22, 19, 22, 22, 20]

H3_2px = [49, 47, 48, 49, 47, 42, 47, 51, 47, 49, 48, 48, 49, 48, 48, 47, 47, 51, 48]
H7_2px = [40, 48, 50, 44, 50, 49, 51, 46, 46, 51, 46, 51, 42, 43, 38, 42, 43, 52, 47]

H3_2py = [52, 50, 53, 50, 54, 51, 50, 53, 54, 50, 53, 50, 52, 50, 53, 54, 54, 53, 50]
H7_2py = [50, 52, 51, 48, 53, 50, 53, 50, 48, 53, 47, 55, 46, 53, 42, 44, 45, 54, 49]

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
		f.write(f'{ang*10} {p1} {p2} {m}\n')

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