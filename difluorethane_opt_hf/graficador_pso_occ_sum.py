import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_pso_occ_C2F2H4_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']


occ_lmo = ['H3_1s', 'H7_1s']

occ_lmo = ['F3_2pz','F7_2pz','F3_2p1','F7_2p1',
'F3_2p2','F7_2p2']

#occ_lmo = ['F3_2p1','F7_2p1', 'F3_2p2','F7_2p2','F3_2pz','F7_2pz']

#occ_lmo = ['F3_2s','F7_2s','F3_2pz','F7_2pz','F3_2p1','F7_2p1']

occ_lmo1 = ['F3_2p2', 'F3_2p1']

occ_lmo2 = ['F7_2p2', 'F7_2p1']


df_F_C = data_J[(data_J.a.str.contains('F3_1s') == True) & (data_J.b.str.contains('F7_1s') == True)].reset_index()
df_F_C.fc_ab = 0
          
for orb1 in occ_lmo:
    for orb2 in occ_lmo:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C.fc_ab += df.reset_index().fc_ab
            b = df.b
            fc = df.fc
            fc_ab = df.fc_ab

fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))

ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, fc, 'g<-', label='$^{FC}J(F-F)$')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
plt.suptitle('PSO contribution to $^3J(H-H)_{i,j}$ in C$_2$F$_2$H$_4$, cc-pVDZ')
ax1.set_title('i=C-F, j=C-F , a = b = all')# f'a={orb1}, b={orb2}')
#plt.savefig('C2F2H4_pso_occ_2pxy.png')
plt.show()