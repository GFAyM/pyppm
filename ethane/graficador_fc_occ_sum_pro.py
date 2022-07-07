import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_occ_C2H6_ccpvdz.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']


occ_lmo = ['H3_1s', 'H7_1s']

occ_lmo1 = ['H3_1s']
occ_lmo2 = ['H7_1s']

df_F_C = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H7_1s') == True)].reset_index()
df_F_C.fc_ab = 0

for orb1 in occ_lmo1:
    for orb2 in occ_lmo1:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C.fc_ab += df.reset_index().fc_ab
            b = df.b
            fc = df.fc
            fc_ab = df.fc_ab
            print(orb1,orb2)

df_F_C_2 = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H7_1s') == True)].reset_index()
df_F_C_2.fc_ab = 0
for orb1 in occ_lmo2:
    for orb2 in occ_lmo2:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C_2.fc_ab += df.reset_index().fc_ab
            b = df.b
            fc = df.fc
            fc_ab = df.fc_ab
            print(orb1,orb2)

df_F_C_3 = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H7_1s') == True)].reset_index()
df_F_C_3.fc_ab = 0
for orb1 in occ_lmo1:
    for orb2 in occ_lmo2:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C_3.fc_ab += df.reset_index().fc_ab
            b = df.b
            fc = df.fc
            fc_ab = df.fc_ab

df_F_C_4 = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H7_1s') == True)].reset_index()
df_F_C_4.fc_ab = 0
          
for orb1 in occ_lmo:
    for orb2 in occ_lmo:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C_4.fc_ab += df.reset_index().fc_ab
            b = df.b
            fc = df.fc
            fc_ab = df.fc_ab


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))

ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, fc, 'g<-', label='$^{FC}J(F-F)$')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
plt.suptitle('FC contribution to $^3J(H-H)_{i,j}$ with all virtuals LMO in C$_2$H$_6$, cc-pVDZ')
ax1.set_title('i=C-H$_1$, j=C-H$_1$')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)


ax2.set_xlabel('Dihedral angle')
ax2.plot(ang, df_F_C_2.fc_ab, 'b>-', label='$^{FC}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, fc, 'g<-', label='$^{FC}J(F-F)$')
ax2.legend()
ax2.set_title('i=C-H$_2$, j=C-H$_2$')# f'a={orb1}, b={orb2}')

ax3.set_xlabel('Dihedral angle')
ax3.plot(ang, df_F_C_3.fc_ab, 'b>-', label='$^{FC}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, fc, 'g<-', label='$^{FC}J(F-F)$')
ax3.legend()
ax3.set_title('i=C-H$_1$, j=C-H$_2$')# f'a={orb1}, b={orb2}')

ax4.set_xlabel('Dihedral angle')
ax4.plot(ang, df_F_C_4.fc_ab, 'b>-', label='$^{FC}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, fc, 'g<-', label='$^{FC}J(F-F)$')
ax4.legend()
ax4.set_title('i=j=[C-H$_1$;C-H$_2$]')# f'a={orb1}, b={orb2}')


plt.savefig('C2H6_FC_occ.png')
plt.show()    

