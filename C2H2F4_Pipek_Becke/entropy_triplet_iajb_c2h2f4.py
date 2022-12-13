import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.ppe_3 import M_matrix

from src.help_functions import extra_functions
import matplotlib.pyplot as plt
import pandas as pd

text = 'entanglement_triplet_c2f6.txt'
#print('number of threads:',lib.num_threads())
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

#lmo_vir = [(vir1_1s,"H1_1s"),(vir2_1s,"H2_1s"),(vir1_2s,"H1_2s"),
#			(vir2_2s,"H2_2s"), (vir1_2px,"H1_2px"),
#		   (vir2_2px,"H2_2px"), (vir1_2py,"H1_2py"),
#		   (vir2_2py,"H2_2py"), (vir1_2pz,"H1_2pz"), (vir2_2pz,"H2_2pz")]

#lmo_virt = [
#            (vir1_2px,vir2_2px,"2px"), (vir1_2py,vir2_2py,"2py")]



data = []

for ang in range(0,18,1): 
    mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H2F4_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

    inv_prop = M_matrix(mol=mol, mo_coeff=mo_coeff, mo_occ=mo_occ,
                occ = [  occ1[ang], occ2[ang]],
                vir = [ v1_1[ang], v2_1[ang],v3_1[ang],
                        v1_2[ang], v2_2[ang],v3_2[ang]])
    
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

plt.suptitle(r'''Triplet Quantum Entanglement in C$_2$H$_2$F$_4$ 
using Localized Molecular Orbitals Pipek-Mezey with Becke parcial charge
with 3 anti-ligants in each excitation''')

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
#plt.savefig('entanglement_triplet_c2h4f2_v123.png')
plt.show()
#if os.path.exists('entanglement_triplet_c2f4h2.txt'):
#    os.remove('entanglement_triplet_c2f4h2.txt')