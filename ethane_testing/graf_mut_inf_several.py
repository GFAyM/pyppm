import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_mi = pd.read_csv('mutual_information_C2H6.txt', sep='\s+', header=None)

df_mi.columns = ['ang', 'Mutual', 'ent_ia', 'ent_jb', 'ent_iajb', 'a', 'b']
        		
plt.figure(figsize=(10,8))
plt.plot(df_mi.ang, df_mi.ent_jb, 'bo-', label='$I_{ia,jb}$' )#f'a={orb1} b={orb2}')
#plt.plot(ang, DSO, 'm--', label='DSO')
#plt.plot(ang, df_F_C.fc, 'go-', label='$^{FC}J(H-H)$')
#plt.plot(ang, FCSD+FC+PSO, 'm--', label='Total')
#plt.plot(ang, FCSD, 'r+-', label='FC+SD')

plt.legend()
plt.ylabel('Mutual Information')
plt.xlabel('Ángulo diedro')
plt.suptitle('Triplet Mutual Information $I_{ia,jb}$ C$_2$H$_6$, cc-pVDZ')
plt.title('i=O-H$_1$, a=O-H$_1$*(1s,2s,2px,2pz,2py), j=O-H$_2$, b= O-H$_2$*(1s,2s,2px,2pz,2py)')# f'a={orb1}, b={orb2}')
#plt.set_size_inches(6.5, 6.5)
#plt.savefig(f'Mutual_inf_C2H6_5exc.png', dpi=200)
plt.show()