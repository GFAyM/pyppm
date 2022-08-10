import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_pso_occ_C2F2H4_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']



occ_lmo = ['F3_1s','F7_1s','F3_2s','F7_2s','F3_2pz','F7_2pz','F3_2p1','F7_2p1',
'F3_2p2','F7_2p2','C1_1s','C2_1s','C_C']

#occ_lmo = ['F3_2p1','F7_2p1', 'F3_2p2','F7_2p2']#, 'F3_2pz', 'F7_2pz']
occ_lmo = ['F3_2pz', 'F7_2pz']



occ_lmo1 = ['F3_2p2', 'F3_2p1']#, 'F3_2pz']
occ_lmo1 = [ 'F3_2pz']


occ_lmo2 = ['F7_2p2', 'F7_2p1']#, 'F7_2pz']
occ_lmo2 = ['F7_2pz']


df_F_C = data_J[(data_J.a.str.contains('F3_1s') == True) & (data_J.b.str.contains('F7_1s') == True)].reset_index()
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

df_F_C_2 = data_J[(data_J.a.str.contains('F3_1s') == True) & (data_J.b.str.contains('F7_1s') == True)].reset_index()
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

df_F_C_3 = data_J[(data_J.a.str.contains('F3_1s') == True) & (data_J.b.str.contains('F7_1s') == True)].reset_index()
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


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,8))

ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{PSO}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
#ax1.plot(ang, fc, 'g<-', label='$^{PSO}J(F-F)$')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
plt.suptitle('PSO contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
ax1.set_title('i=C-F$_1$(2p$_x$,2p$_y$), j=C-F$_1$(2p$_x$,2p$_y$)')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)


ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
ax2.plot(ang, df_F_C_2.fc_ab, 'b>-', label='$^{PSO}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
#ax2.plot(ang, fc, 'g<-', label='$^{PSO}J(F-F)$')
ax2.legend()
ax2.set_title('i=C-F$_2$(2p$_x$,2p$_y$), j=C-F$_2$(2p$_x$,2p$_y$)')# f'a={orb1}, b={orb2}')

ax3.set_ylabel('Hz')
ax3.set_xlabel('Dihedral angle')
ax3.plot(ang, df_F_C_3.fc_ab, 'b>-', label='$^{PSO}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
#ax3.plot(ang, fc, 'g<-', label='$^{PSO}J(F-F)$')
ax3.legend()
ax3.set_title('i=C-F$_1$(2p$_x$,2p$_y$), j=C-F$_2$(2p$_x$,2p$_y$)')# f'a={orb1}, b={orb2}')

#plt.savefig('C2F2H4_pso_occ_2px_2py_2pz.png')
plt.show()    

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
#ax1.plot(ang, fc, 'g<-', label='$^{FC}J(F-F)$')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
plt.suptitle('PSO contribution to $^3J(H-H)_{i,j}$ in C$_2$F$_2$H$_4$, cc-pVDZ')
ax1.set_title('i=C-F, j=C-F , a = b = all')# f'a={orb1}, b={orb2}')
#plt.savefig('C2F2H4_pso_occ_2pxyz.png')
plt.show()