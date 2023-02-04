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

text = 'mutual_information_H2O2_new.txt'
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

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff

    occ = [H3_1s_occ[ang],H4_1s_occ[ang]]
    vir = [H3_1s[ang],H3_2s[ang],H3_2px[ang],H3_2pz[ang],
           H4_1s[ang],H4_2s[ang],H4_2px[ang],H4_2pz[ang]]
    m_obj = M_matrix(occ=occ, vir=vir, mo_coeff=mo_coeff_loc, mol=mol_loc, mo_occ=mo_occ_loc)
    ent_iajb = m_obj.entropy_iajb  
    ent_ia = m_obj.entropy_iaia
    ent_jb = m_obj.entropy_jbjb
    mutual = -ent_ia - ent_jb + ent_iajb
            #print(cruzada)
    with open(text, 'a') as f:
        f.write(f'{ang*10} {ent_ia} {ent_iajb} {ent_jb} {mutual} \n') #{np.round(cruzada, decimals=10)}

df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang', 'ent_iaia', 'ent_iajb', 'ent_jbjb', 'mutual']

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(16,8))
#plt.figure(figsize=(10,8))
ax1.plot(df.ang, df.ent_iaia, 'b>-', label='iaia') #f'a={orb1} b={orb2}')
ax1.set_title(r'$S_{ia}$')
ax1.legend()
ax2.plot(df.ang, df.ent_jbjb, 'b>-', label='jbjb') #f'a={orb1} b={orb2}')
ax2.set_title(r'$S_{jb}$')
ax2.legend()
ax3.plot(df.ang, df.ent_iajb, 'b>-', label='iajb') #f'a={orb1} b={orb2}')
ax3.set_title(r'$S_{iajb}$')
ax3.legend()
ax4.plot(df.ang, df.ent_iaia, 'b>-', label='Mutual Information') #f'a={orb1} b={orb2}')
ax4.set_title(r'Mutual Information')
ax4.legend()

plt.savefig('mutual_information.png')
plt.show()  