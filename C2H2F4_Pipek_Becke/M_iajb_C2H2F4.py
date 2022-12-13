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
from scipy.linalg import expm

#print('number of threads:',lib.num_threads())
text = str('m_p_C2H2F4_iajb.txt')
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

	m = cloppa_obj.M(triplet=True, energy_m=True)
	p = np.linalg.inv(m)
	m = expm(m)
	p = expm(p)
	m = m.reshape(cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir)
	p = -p.reshape(cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir)

	p_iajb_3 = p[occ1[ang], v3_1[ang]-cloppa_obj.nocc, occ2[ang], v3_2[ang]-cloppa_obj.nocc]
	m_iajb_3 = m[occ1[ang], v3_1[ang]-cloppa_obj.nocc, occ2[ang], v3_2[ang]-cloppa_obj.nocc]
	p_iajb_1 = p[occ1[ang], v1_1[ang]-cloppa_obj.nocc, occ2[ang], v1_2[ang]-cloppa_obj.nocc]
	m_iajb_1 = m[occ1[ang], v1_1[ang]-cloppa_obj.nocc, occ2[ang], v1_2[ang]-cloppa_obj.nocc]
	
	with open(text, 'a') as f:
		f.write(f'{ang*10} {m_iajb_1} {p_iajb_1} {m_iajb_3} {p_iajb_3}\n')

#					print(p1, m, p2)

df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang','m_iajb_2','p_iajb_2', 'm_iajb_3', 'p_iajb_3']

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(18,8))
#plt.figure(figsize=(10,8))
ax1.plot(df.ang, df.m_iajb_2, 'b>-', label='M') #f'a={orb1} b={orb2}')
ax1.set_title(r'$M_{iajb}$')
ax1.legend()
ax2.plot(df.ang, df.p_iajb_2, 'b>-', label='P') #f'a={orb1} b={orb2}')
ax2.set_title(r'$P_{iajb}$')
ax2.legend()
ax3.plot(df.ang, df.m_iajb_3, 'b>-', label='P') #f'a={orb1} b={orb2}')
ax3.set_title(r'$M_{iajb_2}$')
ax3.legend()
ax4.plot(df.ang, df.p_iajb_3, 'b>-', label='P') #f'a={orb1} b={orb2}')
ax4.set_title(r'$P_{iajb_2}$')
ax4.legend()
#plt.ylabel('Hz')
ax1.set_xlabel('Ángulo diedro')
plt.suptitle(r'''Elements of Polarization Propagator $J^{FC}(H-H)$ in C$_2$H$_2$F$_4$''')
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
plt.savefig('M_C2H2F4_CH1_CH1*_CH2_CH2**_.png', dpi=200)
plt.show()  