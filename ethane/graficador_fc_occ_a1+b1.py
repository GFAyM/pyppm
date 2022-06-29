import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_a1+b1_occ_C2H6_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']

#####PARA ANALIZAR
#data_J[(data_J.LMOS.str.contains('F3') == True) & (data_J.pp > 1)]
#data_J.LMOS +'*'+ data_J.LMOS
#orb = ['F3_2pz']
#for a in []

occ_lmo = ['H3_1s', 'H7_1s']

occ_lmo1 = ['H3_1s']
occ_lmo2 = ['H7_1s']

df_F_C = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H7_1s') == True)].reset_index()
df_F_C.fc_ab = 0

#print(df_F_C)

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

df_F_C_2 = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H7_1s') == True)].reset_index()
df_F_C_2.fc_ab = 0
for orb1 in occ_lmo1:
    for orb2 in occ_lmo2:
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C_2.fc_ab += df.reset_index().fc_ab
            b = df.b
            fc = df.fc
            fc_ab = df.fc_ab


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))

#fig.figure(figsize=(10,8))
ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')
ax1.set_title('i=j=[C-H$_1$,C-H$_2$]')
ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral Angle')
ax2.plot(ang, df_F_C_2.fc_ab, 'ro-', label='$^{FC}J_{ij}(H-H)$' )
ax2.set_title('i=C-H$_1$  ;   j=C-H$_2$,  a=b=all')
ax2.legend()

fig.suptitle('FC (A(1)+B(1)) contribution to $^3J(H-H)_{i,j}$ en C$_2$O$_6$, cc-pVDZ')

#fig.ylabel('Hz')
#fig.xlabel('Ángulo diedro')
#plt.title('i=j=C-H, a = b = all')# f'a={orb1}, b={orb2}')
plt.savefig(f'FC_a1+b1_occ_C2H6.png', dpi=200)
plt.show()                #


