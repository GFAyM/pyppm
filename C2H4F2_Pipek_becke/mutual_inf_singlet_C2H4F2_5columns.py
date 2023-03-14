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

text = 'mutual_information_difluoro.txt'
if os.path.exists(text):
    os.remove(text)



lig1 = [9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
lig2 = [8, 8, 8, 8, 8, 7, 8, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8]

par_lib_1 = [10, 11, 10, 10, 10, 11, 11, 11, 10, 11, 11, 10, 11, 10, 11, 11, 11, 10, 11]
par_lib_2 = [11, 10, 11, 11, 11, 10, 10, 10, 11, 10, 10, 11, 10, 11, 10, 10, 10, 11, 10]

par_libx_1 = [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5]
par_libx_2 = [4, 5, 6, 6, 6, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5, 4]

par_liby_1 = [7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7]
par_liby_2 = [6, 7, 7, 7, 7, 8, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6]
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
#occ_lmo = [(occ1,'O-H1'), (occ2,'O-H2')]

v1_1 = [31, 74, 74, 73, 73, 73, 73, 73, 74, 74, 74, 73, 73, 73, 73, 73, 73, 73, 74]
v1_2 = [73, 31, 31, 31, 31, 74, 74, 74, 30, 30, 30, 74, 74, 74, 31, 31, 31, 31, 31]

v2_1 = [29, 50, 39, 40, 40, 40, 39, 40, 32, 31, 32, 33, 33, 40, 40, 40, 40, 41, 40]
v2_2 = [49, 29, 29, 23, 25, 23, 22, 32, 31, 32, 31, 22, 23, 23, 25, 29, 22, 29, 29] #mezcla de 3pz y 3s

v3_1 = [41, 32, 32, 32, 32, 33, 33, 33, 40, 41, 40, 40, 39, 33, 32, 32, 32, 32, 32]
v3_2 = [32, 41, 40, 41, 41, 50, 43, 41, 39, 40, 39, 41, 43, 41, 44, 43, 41, 40, 41] #mezcla de 3s y 3pz
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
v4_1 = [45, 40, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 44, 45] #px
v4_2 = [40, 48, 48, 48, 46, 41, 47, 48, 49, 46, 49, 48, 47, 46, 41, 42, 48, 45, 44]

v5_1 = [50, 45, 51, 51, 51, 51, 51, 50, 50, 50, 50, 49, 49, 51, 50, 51, 50, 51, 50] #py
v5_2 = [46, 51, 50, 49, 50, 44, 50, 49, 51, 49, 51, 50, 51, 49, 48, 49, 49, 50, 49]

for ang in range(0,19,1):
    mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H4F2_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

    occ = [lig1[ang],par_liby_1[ang],par_libx_1[ang] ,# , ,    # lig1[ang] ,
           lig2[ang],par_liby_2[ang],par_libx_2[ang]]#, par_libx_2[ang], par_liby_2[ang] ]  #  lig2[ang] ,
    vir = [v1_1[ang],v2_1[ang], v3_1[ang], v4_1[ang], v5_1[ang],  
           v1_2[ang],v2_2[ang], v3_2[ang], v4_2[ang], v5_2[ang]]
    m_obj = M_matrix(occ=occ, vir=vir, mo_coeff=mo_coeff, mol=mol, mo_occ=mo_occ, triplet=False)
    ent_iajb = m_obj.entropy_ab  
    ent_ia = m_obj.entropy_iaia
    ent_jb = m_obj.entropy_jbjb
    ent_iajb_diag0 = m_obj.entropy_iajb
    mutual = ent_ia + ent_jb - ent_iajb
    with open(text, 'a') as f:
        f.write(f'{ang*10} {ent_ia} {ent_iajb} {ent_jb} {mutual} {ent_iajb_diag0} \n') 

df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang', 'ent_iaia', 'ent_iajb', 'ent_jbjb', 'mutual', 'diag0']

fig, (ax1,ax2,ax3,ax4, ax5) = plt.subplots(1, 5, figsize=(24,8))
#plt.figure(figsize=(10,8))
ax1.plot(df.ang, df.ent_iaia, 'b>-', label=r'$S_{ia}(1)$') #f'a={orb1} b={orb2}')
#ax1.set_title(r'$S_{ia}(1)$')
ax1.set_xlabel('Dihedral angle')
ax1.legend()
ax2.plot(df.ang, df.ent_jbjb, 'b>-', label=r'$S_{jb}(1)$') #f'a={orb1} b={orb2}')
#ax2.set_title(r'$S_{jb}(1)$')
ax2.set_xlabel('Dihedral angle')
ax2.legend()
ax3.plot(df.ang, df.ent_iajb, 'b>-', label=r'$S_{iajb}(2)$') #f'a={orb1} b={orb2}')
#ax3.set_title(r'$S_{iajb}(2)$')
ax3.set_xlabel('Dihedral angle')
ax3.legend()
ax4.plot(df.ang, df.mutual, 'b>-', label=r'Mutual Information') #f'a={orb1} b={orb2}')
ax4.set_xlabel('Dihedral angle')
ax4.set_title(r'$S_{ia}(1)$+$S_{ij}(1)$-$S_{ia,jb}(2)$')
ax4.legend()
ax5.plot(df.ang, df.diag0, 'b>-', label=r'Diag=0') #f'a={orb1} b={orb2}')
ax5.set_xlabel('Dihedral angle')
ax5.set_title(r'$S_{ia,jb}(2)$ ')
ax5.legend()

plt.suptitle(r'Medidas de entrelazamiento cuántico singlete utilizando Ligantes y PL 2pxy ')#, 2p$_y$ y 2p$_z$')
plt.savefig('C2H4F2_entanglement_singlet_ligant_PL_pxy.png')
plt.show()  