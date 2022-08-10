import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.ppe import inverse_principal_propagator
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import ao2mo
import itertools

if os.path.exists('mutual_information_H2O2.txt'):
    os.remove('mutual_information_H2O2.txt')



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

vir_lmo = [(H3_1s, 'H3_1s'), (H4_1s, 'H4_1s'), (H3_2s, 'H3_2s'), (H4_2s, 'H4_2s'), 
			  (H3_2px, 'H3_2px'), (H4_2px, 'H4_2px'), (H3_2py, 'H3_2py'), (H4_2py, 'H4_2py'), 
				(H3_2pz, 'H3_2pz'),(H4_2pz, 'H4_2pz'), 
			    (O1_3dz, 'O1_3dz'),(O2_3dz, 'O2_3dz'),
				(O1_3s, 'O1_3s'), (O2_3s, 'O2_3s'), (O1_3dx,'O1_3dx'), (O2_3dx, 'O2_3dx'),
				(O1_3py, 'O1_3py'), (O2_3py, 'O2_3py'),(O1_3dy,'O1_3dy'),(O2_3dy,'O2_3dy'),#]#,
				(O1_3dxz,'O1_3ddxz'),(O2_3dxz,'O2_3ddxz')]
for ang in range(1,2,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff

    i = [H3_1s[ang],H3_2s[ang]]
    #i = [H3_1s[ang],H3_2s[ang],H3_2px[ang],H3_2pz[ang]]

    j = [H4_1s[ang],H4_2s[ang]]
    #j = [H4_1s[ang],H4_2s[ang],H4_2px[ang],H4_2pz[ang]]

    m_obj = inverse_principal_propagator(o1=[H3_1s_occ[ang]], o2=[H4_1s_occ[ang]], v1=i, v2=j, mo_coeff=mo_coeff_loc, mol=mol_loc)
    #cruzada = m_obj.entropy_ia
    print(i,j, m_obj.o1,m_obj.o2)
    print(m_obj.m_iajb_mixedstate)
            #print(cruzada)
    #with open('mutual_information_H2O2.txt', 'a') as f:
    #    f.write(f'{ang*10} {np.round(cruzada, decimals=6)} \n')

#df_mi = pd.read_csv('mutual_information_H2O2.txt', sep='\s+', header=None)

#df_mi.columns = ['ang', 'Mutual']
        		
#plt.figure(figsize=(10,8))
#plt.plot(df_mi.ang, df_mi.Mutual, 'bo-', label='$I_{ia,jb}$' )#f'a={orb1} b={orb2}')
#plt.plot(ang, DSO, 'm--', label='DSO')
#plt.plot(ang, df_F_C.fc, 'go-', label='$^{FC}J(H-H)$')
#plt.plot(ang, FCSD+FC+PSO, 'm--', label='Total')
#plt.plot(ang, FCSD, 'r+-', label='FC+SD')

#plt.legend()
#plt.ylabel('Mutual Information')
#plt.xlabel('Ángulo diedro')
#plt.suptitle('Triplet Mutual Information $I_{ia,jb}$ H$_2$O$_2$, 6-31G**')
#plt.title('i=O-H$_1$, a=O-H$_1$*(1s,2s,2px,2pz,2py), j=O-H$_2$, b= O-H$_2$*(1s,2s,2px,2pz,2py)')# f'a={orb1}, b={orb2}')
#plt.set_size_inches(6.5, 6.5)
#plt.savefig(f'Mutual_inf_H2O2_5exc.png', dpi=200)
#plt.show()