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




for ang in range(0,18,1):
	mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
	cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, #vir=viridx, occ=occidx,
	mo_occ_loc=mo_occ)

	m = cloppa_obj.M(triplet=True, energy_m=False)
	p = np.linalg.inv(m)
	m = m.reshape(cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir)
	p = -p.reshape(cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir)


	p_iajb_3 = p[H3_1s_occ[ang], H3_1s[ang]-cloppa_obj.nocc, H7_1s_occ[ang], H7_1s[ang]-cloppa_obj.nocc]
	m_iajb_3 = m[H3_1s_occ[ang], H3_1s[ang]-cloppa_obj.nocc, H7_1s_occ[ang], H7_1s[ang]-cloppa_obj.nocc]
	p_iajb_2 = p[H3_1s_occ[ang], H3_2pz[ang]-cloppa_obj.nocc, H7_1s_occ[ang], H7_2pz[ang]-cloppa_obj.nocc]
	m_iajb_2 = m[H3_1s_occ[ang], H3_2pz[ang]-cloppa_obj.nocc, H7_1s_occ[ang], H7_2pz[ang]-cloppa_obj.nocc]
	
#	p = p.reshape(cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir)
#	p_iajb = p[occ1[ang],v1_1[ang] - cloppa_obj.nocc,occ2[ang],v1_2[ang] - cloppa_obj.nocc]
	with open(text, 'a') as f:
		f.write(f'{ang*10} {abs(m_iajb_2)} {abs(p_iajb_2)} {abs(m_iajb_3)} {abs(p_iajb_3)}\n')

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