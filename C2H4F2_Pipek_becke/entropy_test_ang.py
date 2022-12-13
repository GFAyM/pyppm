import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.ppe import M_matrix
import matplotlib.pyplot as plt
import pandas as pd


#print('number of threads:',lib.num_threads())
text = str('ent_test.txt')
if os.path.exists(text):
	os.remove(text)


lig1 = [9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
lig2 = [8, 8, 8, 8, 8, 7, 8, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8]

par_lib_1 = [10, 11, 10, 10, 10, 11, 11, 11, 10, 11, 11, 10, 11, 10, 11, 11, 11, 10, 11]
par_lib_2 = [11, 10, 11, 11, 11, 10, 10, 10, 11, 10, 10, 11, 10, 11, 10, 10, 10, 11, 10]

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
#       10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27

#lmo_vir = [(vir1_1s,"H1_1s"),(vir2_1s,"H2_1s"),(vir1_2s,"H1_2s"),
#			(vir2_2s,"H2_2s"), (vir1_2px,"H1_2px"),
#		   (vir2_2px,"H2_2px"), (vir1_2py,"H1_2py"),
#		   (vir2_2py,"H2_2py"), (vir1_2pz,"H1_2pz"), (vir2_2pz,"H2_2pz")]

#lmo_virt = [
#            (vir1_2px,vir2_2px,"2px"), (vir1_2py,vir2_2py,"2py")]


ang = 0
for ang in range(0,18,1):
	mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H4F2_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

	inv_prop = M_matrix(mol=mol, mo_coeff=mo_coeff, mo_occ=mo_occ,
				occ = [  lig1[ang], lig2[ang]],
				vir = [ v1_1[ang],v3_1[ang],v2_1[ang], v4_1[ang],v5_1[ang],
						v1_2[ang],v3_2[ang],v2_2[ang], v4_2[ang],v5_2[ang]])   
	ent_ia = inv_prop.entropy_ia
	ent_iajb = inv_prop.entropy_iajb
	ent_jb = inv_prop.entropy_jb
	mutual = ent_ia + ent_jb - ent_iajb
	with open(text, 'a') as f:
		f.write(f'{ang*10} {ent_ia} {ent_jb} {ent_iajb} {mutual} \n')


data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'ent_ia', 'ent_jb', 'ent_iajb', 'mutual']

fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(18,8))

ax1.plot(data_J.ang, data_J.ent_ia, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')

plt.suptitle(r'''Triplet Quantum Entanglement in C$_2$H$_4$F$_2$ 
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
plt.savefig('entanglement_triplet_c2h4f2_new_tests.png')
plt.show()