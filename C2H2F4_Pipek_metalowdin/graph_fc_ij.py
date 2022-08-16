import matplotlib.pyplot as plt
import pandas as pd


data_J = pd.read_csv('cloppa_fc_ij_C2H4F2.txt', sep='\s+', header=None)
data_J = pd.DataFrame(data_J)


ang = data_J[0]
FC = data_J[1]
FC_ij = data_J[2]

#plt.plot(ang, DSO, 'ro', label='DSO')
plt.figure(figsize=(9,10))

plt.plot(ang, FC, 'go-', label=r'J$^{FC}_{Total}$')
plt.plot(ang, FC_ij, 'mo-', label=r'J$^{FC}_{ij}$')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Dihedral angle')
plt.title(r'J$^{FC}_{ij}(H-H)$ with ligants C-H in C$_2$H$_2$F$_4$, cc-pVDZ')
plt.savefig('FC_occ_C2H2F4_ccpvdz.png', dpi=200)
plt.show()