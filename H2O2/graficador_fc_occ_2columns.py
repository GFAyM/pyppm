import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_occ_H2O2_631G.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']

#####PARA ANALIZAR
#data_J[(data_J.LMOS.str.contains('F3') == True) & (data_J.pp > 1)]
#data_J.LMOS +'*'+ data_J.LMOS
#orb = ['F3_2pz']
#for a in []

occ_lmo = ['O1_1s', 'O2_1s',  'O1_2p1','O2_2p1','O1_2p2', 'O2_2p2','H4_1s','H3_1s','C_C']

#occ_lmo = ['O1_1s', 'O2_1s',  'O1_2p1','O2_2p1','O1_2p2', 'O2_2p2','C_C']

occ_lmo = ['H4_1s','H3_1s' ]

occ_lmo1 = ['H3_1s']
occ_lmo2 = ['H4_1s']



df_F_C = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H4_1s') == True)].reset_index()
df_F_C.fc_ab = 0
for orb1 in occ_lmo:
    for orb2 in occ_lmo:

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C.fc_ab += df.reset_index().fc_ab
            fc = df.fc

df_F_C_2 = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H4_1s') == True)].reset_index()
df_F_C_2.fc_ab = 0
for orb1 in occ_lmo1:
    for orb2 in occ_lmo2:

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C_2.fc_ab += df.reset_index().fc_ab
            fc = df.fc
            

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
##plt.plot(ang, DSO, 'm--', label='DSO')
ax1.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
ax1.set_title('i = Lig1, j = Lig2; a = b = all')# f'a={orb1}, b={orb2}')

ax1.legend()

ax2.plot(ang, df_F_C_2.fc_ab, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
##plt.plot(ang, DSO, 'm--', label='DSO')
ax2.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')
ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
ax2.set_title('i = {Lig1,Lig2}, j = {Lig1,Lig2}; a = b = all')# f'a={orb1}, b={orb2}')
ax2.legend()

plt.suptitle('FC contribution to $^3J(H_1-H_2)_{i,j}$ in H$_2$O$_2$, 6-31G** basis')


plt.savefig(f'FC_occ_H2O2_ij_pathways.png', dpi=200)

plt.show()                #
