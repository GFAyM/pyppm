import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.ppe import M_matrix

from src.help_functions import extra_functions
import matplotlib.pyplot as plt
import pandas as pd


if os.path.exists('entanglement_c2h4f2_singlet.txt'):
    os.remove('entanglement_c2h4f2_singlet.txt')
F3_2py =[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
F7_2py =[5, 5, 7, 7, 7, 7, 7, 7, 5, 5, 5, 6, 7, 7, 7, 7, 6, 5, 5]
    #    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
F3_2px =[7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7]
F7_2px =[6, 7, 6, 6, 6, 6, 6, 6, 7, 6, 7, 7, 6, 6, 6, 6, 7, 7, 6]

F3_2pz = [8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
F7_2pz = [9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]


v1_1= [73, 74, 74, 73, 73, 73, 73, 73, 74, 74, 73, 73, 73, 73, 73, 73, 73, 74, 74]
       #0   1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18 
v1_2= [74, 56, 73, 74, 75, 61, 75, 75, 73, 73, 74, 75, 75, 75, 74, 74, 75, 73, 73]

v2_1= [21, 21, 21, 20, 21, 21, 21, 21, 21, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21]
v2_2= [22, 22, 22, 21, 25, 23, 22, 22, 22, 21, 22, 22, 23, 23, 25, 22, 22, 22, 22]


v3_1= [41, 40, 39, 40, 40, 40, 39, 40, 40, 41, 40, 40, 39, 40, 40, 40, 40, 40, 40]
v3_2= [40, 41, 40, 41, 44, 45, 43, 41, 39, 40, 41, 41, 43, 41, 41, 41, 41, 41, 41]


v4_1= [64, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64]
       #0   1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18
v4_2= [63, 64, 64, 65, 62, 69, 69, 69, 69, 69, 65, 69, 69, 62, 67, 65, 64, 64, 63]

v6_1= [50, 50, 51, 51, 50, 51, 51, 50, 49, 50, 48, 49, 49, 51, 50, 50, 50, 49, 50]
v6_2= [49, 48, 48, 49, 48, 50, 50, 49, 50, 49, 51, 50, 51, 49, 51, 51, 49, 50, 49]


#falta agregar los orbitales 3dy. 
v5_1= [43, 45, 45, 43, 43, 46, 45, 45, 45, 45, 46, 45, 45, 45, 44, 44, 44, 45, 45]
       #0   1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18 
v5_2= [45, 51, 50, 50, 41, 41, 47, 48, 51, 46, 45, 48, 47, 46, 46, 49, 48, 51, 44]

#además, agregar los 3dx y 3dy

v7_1= [68, 67, 67, 67, 68, 66, 67, 68, 67, 68, 68, 68, 67, 68, 66, 67, 68, 67, 68]
v7_2= [67, 69, 69, 69, 65, 68, 68, 67, 66, 65, 66, 67, 68, 66, 69, 69, 70, 69, 67]

v8_1= [69, 71, 71, 71, 70, 71, 71, 71, 71, 70, 70, 71, 71, 70, 71, 71, 71, 71, 70]
v8_2= [70, 70, 70, 70, 66, 67, 65, 64, 64, 64, 64, 64, 65, 65, 70, 70, 69, 70, 69]

v9_1= [62, 62, 62, 62, 61, 62, 62, 62, 62, 62, 61, 62, 62, 61, 62, 62, 62, 62, 61]
v9_2= [61, 68, 65, 64, 64, 64, 64, 65, 68, 61, 62, 65, 64, 64, 64, 64, 65, 68, 62]

v10_1= [66, 65, 66, 66, 67, 65, 66, 66, 65, 66, 67, 66, 66, 67, 65, 66, 66, 65, 66]
v10_2= [65, 66, 68, 68, 69, 70, 70, 70, 70, 67, 69, 70, 70, 69, 68, 68, 67, 66, 65]




lmo_vir = [(v1_1,"F3_2pz"),(v1_2,"F7_2pz"),(v2_1,"F3_3pz"),(v2_2,"F7_3pz"), (v3_1,"F3_3s"),(v3_2,"F7_3s"),
			(v4_1,"F3_3dz"),(v4_2,"F7_3dz"), (v5_1,"F3_3py"), (v5_2,"F7_3py"),(v6_1,"F3_3px"),(v6_2,"F7_3px"),
			(v7_1,"F3_3dxy"),(v7_2,"F7_3dxy"), (v8_1,"F3_3dx2-y2"),(v8_2,"F7_3dx2-y2"), 
			(v9_1,"F3_3dyz"),(v9_2,"F7_3dyz"), (v10_1,"F3_3dxz"), (v10_2, "F7_3dxz")]

#los lmos occupados que más contribuyen a J(PSO) son los 2px, 2py, y 2pz(en menor medida)
#y los lmos virtuales que más contribuyen son : 2pz, 3pz, 3s, 3px,3py en menor medida
data = []
for ang in range(0,19,1): 
    mol, mo_coeff, mo_occ = extra_functions(molden_file=f"difluorethane_cc-pvdz_{ang*10}_Cholesky_PM.molden").extraer_coeff
    inv_prop = M_matrix(mol=mol, mo_coeff=mo_coeff, triplet=False, #
                occ = [F3_2px[ang],F3_2py[ang],F3_2pz[ang],F7_2px[ang],F7_2py[ang],F7_2pz[ang]],
                vir = [ v1_1[ang],v5_1[ang],v6_1[ang], 
                        v1_2[ang],v5_2[ang],v6_2[ang]])
    #m = inv_prop.m_iajb
    #inv_prop_old = inverse_principal_propagator(mol=mol, mo_coeff=mo_coeff, o1 = [F3_2pz[ang]], o2 = [F7_2pz[ang]],
                    #v1 = [v1_1[ang], v2_1[ang]], v2 = [v1_2[ang], v2_2[ang]])
    ent_ia = inv_prop.entropy_iaia
    ent_iajb = inv_prop.entropy_iajb
    ent_jb = inv_prop.entropy_jbjb
    mutual = ent_ia + ent_jb - ent_iajb
    with open('entanglement_c2h4f2_singlet.txt', 'a') as f:
        f.write(f'{ang*10} {ent_ia} {ent_jb} {ent_iajb} {mutual} \n')
    

text = 'entanglement_c2h4f2_singlet.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'ent_ia', 'ent_jb', 'ent_iajb', 'mutual']

fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(16,8))

ax1.plot(data_J.ang, data_J.ent_ia, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')

plt.suptitle(r'''Quantum Entanglement between the LMOs 
i$=F3(2p_{xyz})$; a$=F3(2p_z,3p_{xy})$; i$=F7(2p_{xyz})$; b$=F7({2p_z,3p_{xy}})$''')

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
plt.savefig('entanglement_singlet_c2h4f2.png')
plt.show()