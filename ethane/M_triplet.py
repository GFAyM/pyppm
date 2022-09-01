import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.ppe_3 import M_matrix
import plotly.express as px
from src.cloppa import Cloppa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


text = 'entanglement_triplet_c2f6.txt'

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



vir_lmo1 = [(H3_1s, 'H3_1s'), (H3_2s, 'H3_2s'), (H3_2px, 'H3_2px'), (H3_2py, 'H3_2py'), (H3_2pz, 'H3_2pz')]
vir_lmo2 = [(H7_1s, 'H7_1s'), (H7_2s, 'H7_2s'), (H7_2px, 'H7_2px'), (H7_2py, 'H7_2py'), (H7_2pz, 'H7_2pz')]

for ang in range(0,18,1):
	mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
	cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, #vir=viridx, occ=occidx,
	mo_occ_loc=mo_occ)

	m = cloppa_obj.M(triplet=True)
	p = np.linalg.inv(m)
	m = m.reshape(cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir)
	p = -p.reshape(cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir)


	p_iajb_3 = p[H3_1s_occ[ang], H3_1s[ang]-cloppa_obj.nocc, H7_1s_occ[ang], H7_1s[ang]-cloppa_obj.nocc]
	m_iajb_3 = m[H3_1s_occ[ang], H3_1s[ang]-cloppa_obj.nocc, H7_1s_occ[ang], H7_1s[ang]-cloppa_obj.nocc]
	p_iajb_1 = p[H3_1s_occ[ang], H3_2s[ang]-cloppa_obj.nocc, H7_1s_occ[ang], H7_2s[ang]-cloppa_obj.nocc]
	m_iajb_1 = m[H3_1s_occ[ang], H3_2s[ang]-cloppa_obj.nocc, H7_1s_occ[ang], H7_2s[ang]-cloppa_obj.nocc]
	
#	p = p.reshape(cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir)
#	p_iajb = p[occ1[ang],v1_1[ang] - cloppa_obj.nocc,occ2[ang],v1_2[ang] - cloppa_obj.nocc]
	with open(text, 'a') as f:
		f.write(f'{ang*10} {abs(m_iajb_1)} {abs(p_iajb_1)} {abs(m_iajb_3)} {abs(p_iajb_3)}\n')

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
plt.savefig('M_C2H6_CH1_CH1*_CH2_CH2**_.png', dpi=200)
plt.show()  