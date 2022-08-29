import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.ppe import M_matrix
import plotly.express as px
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
#vir_lmo1 = [(H3_1s, 'H3_1s'), (H3_2s, 'H3_2s'), (H3_2px, 'H3_2px'), (H3_2py, 'H3_2py'), (H3_2pz, 'H3_2pz')]
#vir_lmo2 = [(H7_1s, 'H7_1s'), (H7_2s, 'H7_2s'), (H7_2px, 'H7_2px'), (H7_2py, 'H7_2py'), (H7_2pz, 'H7_2pz')]

for ang in range(0,19,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

    m_obj = inv_prop = M_matrix(mol=mol_loc, mo_coeff=mo_coeff_loc, mo_occ=mo_occ_loc,
                occ = [  H3_1s_occ[ang], H7_1s_occ[ang]],
                vir = [ H3_1s[ang], H3_2s[ang], H3_2px[ang], H3_2py[ang],
                        H7_1s[ang], H7_2s[ang], H7_2px[ang], H7_2py[ang]])
                        
    ent_ia = inv_prop.entropy_iaia
    ent_iajb = inv_prop.entropy_iajb
    ent_jb = inv_prop.entropy_jbjb
    mutual = ent_ia + ent_jb - ent_iajb
    with open(text, 'a') as f:
        f.write(f'{ang*10} {ent_ia} {ent_jb} {ent_iajb} {mutual} \n')


data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'ent_ia', 'ent_jb', 'ent_iajb', 'mutual']

fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(18,8))

ax1.plot(data_J.ang, data_J.ent_ia, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')

plt.suptitle(r'''Triplet Quantum Entanglement in C$_2$H$_6$ 
using Localized Molecular Orbitals Pipek-Mezey ''')

ax1.set_xlabel('Dihedral angle')
ax1.set_ylabel('Entanglement')
ax1.set_title('S$_{ia}$')# f'a={orb1}, b={orb2}')
#i$=$F3$_{2s}$,F3$_{2pz}$ a$=F3$_{3s}$F3$_{2pz}$, j$=$F7$_{2s}$,F7$_{2pz},b$=F7$_{3s}$F7$_{2pz}$

ax2.set_xlabel('Dihedral angle')
ax2.plot(data_J.ang, data_J.ent_jb, 'b>-', label='$^{FC}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax2.set_title('S$_{jb}$')# f'a={orb1}, b={orb2}')

ax3.set_xlabel('Dihedral angle')
ax3.plot(data_J.ang, data_J.ent_iajb, 'b>-', label='$^{FC}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax3.set_title('S$_{iajb}$')# f'a={orb1}, b={orb2}')

ax4.set_xlabel('Dihedral angle')
ax4.plot(data_J.ang, data_J.mutual, 'b>-', label='$^{FC}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax4.set_title('Mutual Information ')# f'a={orb1}, b={orb2}')
#plt.savefig('entanglement_triplet_c2h6_v123.png')
plt.show()