import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.ppe import M_matrix
from src.ppe import inverse_principal_propagator
from src.help_functions import extra_functions
import matplotlib.pyplot as plt
import pandas as pd


#print('number of threads:',lib.num_threads())
if os.path.exists('entanglement_triplet_c2f4h2.txt'):
	os.remove('entanglement_triplet_c2f4h2.txt')


lig_1 = [22, 22, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 23, 22, 23, 23, 23, 22]
lig_2 = [23, 23, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 22, 23, 22, 22, 22, 23]

occ_lmo = [(lig_1,'O-H1'), (lig_2,'O-H2')]

vir1_1s = [63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63]
vir2_1s = [64, 64, 64, 64, 53, 64, 53, 64, 64, 64, 53, 64, 53, 64, 64, 64, 64, 64]

vir1_2s = [36, 36, 36, 36, 36, 35, 35, 36, 36, 36, 35, 35, 36, 36, 36, 36, 36, 36]
vir2_2s = [35, 35, 35, 35, 35, 34, 34, 35, 35, 35, 34, 34, 35, 35, 35, 35, 35, 35]

vir1_2px = [66, 67, 67, 67, 67, 67, 67, 65, 65, 65, 67, 67, 67, 67, 67, 67, 66, 65]
vir2_2px = [65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 65, 65, 66]

vir1_2py = [68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 67]

vir1_2pz = [38, 38, 38, 39, 39, 39, 39, 39, 38, 39, 39, 39, 39, 39, 38, 38, 38, 37]
#           10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
vir2_2py = [67, 66, 65, 65, 65, 65, 65, 67, 67, 67, 65, 65, 65, 38, 65, 66, 67, 68]
vir2_2pz = [37, 37, 37, 38, 64, 38, 38, 38, 39, 38, 38, 38, 64, 65, 37, 37, 37, 38]

lmo_vir = [(vir1_1s,"H1_1s"),(vir2_1s,"H2_1s"),(vir1_2s,"H1_2s"),
			(vir2_2s,"H2_2s"), (vir1_2px,"H1_2px"),
		   (vir2_2px,"H2_2px"), (vir1_2py,"H1_2py"),
		   (vir2_2py,"H2_2py"), (vir1_2pz,"H1_2pz"), (vir2_2pz,"H2_2pz")]


data = []
for ang in range(10,28,1): 
    mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H2F2_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff
    
    vir1 = [ vir1_1s[ang-10],vir1_2pz[ang-10]]
    vir2 = [ vir2_1s[ang-10],vir2_2pz[ang-10]]

    m_obj = inverse_principal_propagator(o1=[lig_1[ang-10]], o2=[lig_2[ang-10]], v1=vir1, v2=vir2, 
             mo_coeff=mo_coeff, mol=mol)
    I = m_obj.mutual_information

    ent_ia = m_obj.entropy_ia
    ent_jb = m_obj.entropy_jb
    ent_iajb_2 = m_obj.entropy_iajb_mixedstate
#    m_iajb = m_obj.m_iajb
    ent_iajb = m_obj.entropy_iajb

    
    mutual = ent_ia + ent_jb - ent_iajb
    with open('entanglement_triplet_c2f4h2.txt', 'a') as f:
        f.write(f'{ang*10} {ent_ia} {ent_jb} {ent_iajb} {mutual} \n')
    

text = 'entanglement_triplet_c2f4h2.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'ent_ia', 'ent_jb', 'ent_iajb', 'mutual']

fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(16,8))

ax1.plot(data_J.ang, data_J.ent_ia, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')

plt.suptitle(r'''Triplet Quantum Entanglement ''')

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
ax4.set_title('Mutual Information')# f'a={orb1}, b={orb2}')
plt.savefig('entanglement_triplet_c2h4f2_1s2pz_oldway.png')      
plt.show()
