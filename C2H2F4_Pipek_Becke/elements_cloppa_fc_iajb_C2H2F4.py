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
text = str('elementos_cloppa__C2H2F4_fc_iajb.txt')
if os.path.exists(text):
	os.remove(text)

occ1 = [23, 22, 22, 22, 23, 23, 23, 22, 23, 22, 23, 22, 22, 22, 23, 22, 23, 23, 23, 22, 22, 23, 23, 23, 23, 22, 22, 23]
occ2 = [22, 23, 23, 23, 22, 22, 22, 23, 22, 23, 22, 23, 23, 23, 22, 23, 22, 22, 22, 23, 23, 22, 22, 22, 22, 23, 23, 22]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
occ_lmo = [(occ1,'O-H1'), (occ2,'O-H2')]

v1_1 = [34, 35, 35, 35, 36, 36, 36, 36, 36, 35, 36, 36, 36, 36, 36, 35, 35, 36, 36, 36, 35, 35, 36, 36, 36, 36, 36, 35]
v1_2 = [35, 34, 34, 34, 35, 35, 35, 35, 35, 36, 35, 35, 35, 35, 35, 34, 34, 35, 35, 35, 34, 34, 35, 35, 35, 35, 35, 36]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27

v2_1 = [64, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 63, 63, 63, 63, 63, 63, 63, 63, 64]
v2_2 = [63, 64, 64, 64, 66, 66, 64, 64, 64, 64, 64, 64, 64, 64, 66, 64, 64, 64, 63, 64, 64, 64, 66, 64, 64, 64, 64, 63]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
v3_1 = [92, 39, 39, 39, 39, 38, 38, 38, 38, 92, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 38]
v3_2 = [39, 92, 92, 92, 92, 92, 92, 93, 93, 38, 92, 92, 92, 92, 92, 91, 91, 92, 92, 92, 91, 91, 92, 92, 92, 92, 92, 92]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
v4_1 = [66, 66, 66, 67, 67, 67, 66, 66, 65, 65, 66, 67, 67, 67, 67, 67, 67, 65, 66, 65, 67, 67, 67, 67, 67, 67, 66, 66]
v4_2 = [65, 65, 65, 65, 64, 64, 65, 65, 66, 66, 65, 65, 65, 65, 64, 65, 65, 66, 65, 66, 65, 65, 64, 65, 65, 65, 65, 65]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
v5_1 = [68, 68, 68, 68, 68, 68, 68, 68, 68, 67, 68, 68, 68, 68, 68, 68, 68, 68, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68]
v5_2 = [67, 67, 67, 66, 65, 65, 67, 67, 67, 68, 67, 66, 66, 66, 65, 66, 66, 67, 68, 67, 66, 66, 65, 66, 66, 66, 67, 67]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
v6_1 = [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]
v6_2 = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 93, 93, 28, 27, 28, 93, 93, 25, 25, 25, 25, 25, 25]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27




for ang in range(0,18,1):
	mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H2F4_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
	cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, #vir=viridx, occ=occidx,
	mo_occ_loc=mo_occ)

	m = cloppa_obj.M(triplet=True)

	p = np.linalg.inv(m)

	p1, m, p2 = cloppa_obj.kernel_pathway_elements(FC=True, FCSD=False, PSO=False,
							princ_prop=p,
							n_atom1=[2], occ_atom1=occ1[ang], vir_atom1=v2_1[ang],
							n_atom2=[6], occ_atom2=occ1[ang], vir_atom2=v2_1[ang], elements=True)

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
plt.suptitle(r'''Elements of Polarization Propagator $J^{FC}(H-H)$ in C$_2$H$_2$F$_4$''')
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
plt.savefig('FC_elements_C2H2F4_CH1_CH1**_CH1_CH1**.png', dpi=200)
plt.show()  