import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.ppe import M_matrix

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


lmo_vir1 = [(v1_1,"H1_2pz"),(v2_1,"H1_1s"),(v3_1,"H1_2s"),
			(v4_1,"H1_2px"),(v5_1,"H1_2py"), (v6_1,"C-H_2") ]


data = []

for ang in range(0,18,1): 
    mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H2F4_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

    inv_prop = M_matrix(mol=mol, mo_coeff=mo_coeff, mo_occ=mo_occ,
                occ = [  occ1[ang], occ2[ang]],
                vir = [ v6_1[ang], v5_1[ang],v3_1[ang],#v5_1[ang],v4_1[ang], #,, 
                        v6_2[ang], v5_2[ang],v3_2[ang]])#v5_2[ang],v4_2[ang]]) #,,  
    ent_iajb = inv_prop.entropy_ab
    ent_ia = inv_prop.entropy_iaia
    ent_iajb_diag0 = inv_prop.entropy_iajb
    ent_jb = inv_prop.entropy_jbjb
    

    mutual = ent_ia + ent_jb - ent_iajb
    with open(text, 'a') as f:
        f.write(f'{ang*10} {ent_ia} {ent_jb} {ent_iajb} {mutual} {ent_iajb_diag0} \n')






data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'ent_ia', 'ent_jb', 'ent_iajb', 'mutual', 'ent_iajb_diag0']

plt.figure(figsize=(12,10))
plt.plot(data_J.ang, data_J.mutual, 'b>-', label='ia') #f'a={orb1} b={orb2}')
#plt.title(r'$S_{ia}(1)$')
plt.xlabel('Dihedral angle')
plt.ylabel('Entanglement')
#ax1.legend()
plt.title(r'''Mutual Information between excitations that contribute the most in $^{FC}J(F-F)$ in C$_2$H$_2$F$_4$''')#, 2p$_y$ y 2p$_z$')


plt.savefig('mutual_triplet_C2H2F4.png')
plt.show()  