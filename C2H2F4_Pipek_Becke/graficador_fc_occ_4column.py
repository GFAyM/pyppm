import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'pathways_fc_c2h2f4.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang','fc_tot', 'fc', 'a', 'b']

occ_lmo = ['O-H1','O-H2']
occ_lmo_1 = ['O-H1']
occ_lmo_2 = ['O-H2']


df_F_C = data_J[(data_J.a.str.contains('O-H1') == True) & (data_J.b.str.contains('O-H1') == True)].reset_index()
df_F_C.fc = 0

for orb1 in occ_lmo:
    for orb2 in occ_lmo:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C.fc += df.reset_index().fc
        
df_F_C_1 = data_J[(data_J.a.str.contains('O-H1') == True) & (data_J.b.str.contains('O-H1') == True)].reset_index()
df_F_C_1.fc = 0
for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_2:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_1.fc += df.reset_index().fc
        
df_F_C_2 = data_J[(data_J.a.str.contains('O-H1') == True) & (data_J.b.str.contains('O-H1') == True)].reset_index()
df_F_C_2.fc = 0
for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_1:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_2.fc += df.reset_index().fc

df_F_C_3 = data_J[(data_J.a.str.contains('O-H1') == True) & (data_J.b.str.contains('O-H1') == True)].reset_index()
df_F_C_3.fc = 0
for orb1 in occ_lmo_2:
    for orb2 in occ_lmo_2:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_3.fc += df.reset_index().fc


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))

ax1.plot(ang, df_F_C.fc, 'b>-', label='$^{fc}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, df_F_C.fc_tot, 'g<-', label='$^{fc}J(H-H)$')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
plt.suptitle('Ligant contributions to $^{FC}J(H-H)_{i,j}$ in C$_2$F$_4$H$_2$')
ax1.set_title(r'i=j=[C-H1, C-H2]')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)


#ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
ax2.plot(ang, df_F_C_1.fc, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, df_F_C.fc_tot, 'g<-', label='$^{FC}J(H-H)$')
ax2.legend()
ax2.set_title(r'i=C-H1, j=C-H2')# f'a={orb1}, b={orb2}')

#ax3.set_ylabel('Hz')
ax3.set_xlabel('Dihedral angle')
ax3.plot(ang, df_F_C_2.fc, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, df_F_C.fc_tot, 'g<-', label='$^{FC}J(H-H)$')
ax3.legend()
ax3.set_title(r'i=j=C-H1')# f'a={orb1}, b={orb2}')

ax4.set_xlabel('Dihedral angle')
ax4.plot(ang, df_F_C_3.fc, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, df_F_C.fc_tot, 'g<-', label='$^{FC}J(H-H)$')
ax4.legend()
ax4.set_title(r'i=j=C-H1')# f'a={orb1}, b={orb2}')

plt.savefig('FC_ij_C2H2F4_ccpvdz.png')
plt.show()    

