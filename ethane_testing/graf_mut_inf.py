import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_mi = pd.read_csv('mutual_information_C2H6.txt', sep='\s+', header=None)

df_mi.columns = ['ang', 'Mutual', 'ent_ia', 'ent_jb', 'ent_iajb_2', 'm_iajb', 'ent_iajb']
        		
plt.figure(figsize=(12,8))
plt.plot(df_mi.ang, np.around(df_mi.Mutual,decimals=6), 'bo-', label='$I_{ia,jb}$' )#f'a={orb1} b={orb2}')

plt.legend()
plt.ylabel('Mutual Information')
plt.xlabel('Dihedral angle')
plt.suptitle('Triplet Mutual Information $I_{ia,jb}$ C$_2$H$_6$, cc-pVDZ')
plt.title('i=O-H$_1$, a=O-H$_1*(1s,2s,2pz)$, j=O-H$_2$, b= O-H$_2*(1s,2s,2p_z)$')# f'a={orb1}, b={orb2}')
plt.savefig(f'Mutual_inf_C2H6.png', dpi=200)
plt.show()