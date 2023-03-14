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
from pyscf import ao2mo
import itertools

text = 'mutual_information_ethane.txt'
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
for ang in range(0,19,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

    occ = [H3_1s_occ[ang],H7_1s_occ[ang]]
    vir = [H3_2s[ang],H3_2pz[ang], H3_2px[ang], H3_2py[ang],
           H7_2s[ang],H7_2pz[ang], H7_2px[ang], H7_2py[ang]]
    m_obj = M_matrix(occ=occ, vir=vir, mo_coeff=mo_coeff_loc, mol=mol_loc, mo_occ=mo_occ_loc)
    ent_iajb = m_obj.entropy_ab  
    ent_ia = m_obj.entropy_iaia
    ent_jb = m_obj.entropy_jbjb
    mutual = ent_ia + ent_jb - ent_iajb
    #print(ent_ia,ent_jb)
            #print(cruzada)
    with open(text, 'a') as f:
        f.write(f'{ang*10} {ent_ia} {ent_iajb} {ent_jb} {mutual} \n') #{np.round(cruzada, decimals=10)}

df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang', 'ent_iaia', 'ent_iajb', 'ent_jbjb', 'mutual']

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(17,8))
#plt.figure(figsize=(10,8))
ax1.plot(df.ang, df.ent_iaia, 'b>-', label='ia') #f'a={orb1} b={orb2}')
ax1.set_title(r'$S_{ia}(1)$')
ax1.set_xlabel('Dihedral angle')
#ax1.legend()
ax2.plot(df.ang, df.ent_jbjb, 'b>-', label='ia') #f'a={orb1} b={orb2}')
ax2.set_title(r'$S_{jb}(1)$')
ax2.set_xlabel('Dihedral angle')
#ax2.legend()
ax3.plot(df.ang, df.ent_iajb, 'b>-', label='ia') #f'a={orb1} b={orb2}')
ax3.set_title(r'$S_{iajb}(2)$')
ax3.set_xlabel('Dihedral angle')
#ax3.legend()
ax4.plot(df.ang, df.ent_iaia, 'b>-', label='ia') #f'a={orb1} b={orb2}')
#ax4.set_title(r'I = $S_{ia}$(1) + S$_{jb}$(1) - S$_{ia,jb}$(2)')
ax4.set_xlabel('Dihedral angle')
ax4.set_title('I')
#ax4.legend()
plt.suptitle('Medidas de entrelazamiento cuántico utilizando los antiligantes 1s, 2s, 2p$_z$')#, 2p$_y$ y 2p$_z$')
plt.savefig('ethane_entanglement_new_normalization.png')
plt.show()  