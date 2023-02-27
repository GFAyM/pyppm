import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_vir_C2H6_ccpvdz.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc_ij']




vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s', 'H3_2px', 'H7_2px','H3_2py', 'H7_2py', 'H3_2pz', 'H7_2pz']
vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s','H3_2pz', 'H7_2pz']

# 'O1_3dz', 'O2_3dz', , 'O1_3py', 'O2_3py', ,'O1_3ddxz','O2_3ddxz']#, 
# 'O1_3dx', 'O2_3dx','O1_3dy', 'O2_3dy','O1_3s', 'O2_3s'
#vir_lmo = ['H3_2s', 'H4_2s', 'H3_2px', 'H4_2px','H3_2py', 'H4_2py', 'H3_2pz', 'H4_2pz','O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']

#'O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']

vir_lmo1 =['H3_1s','H3_2s','H3_2px','H3_2py','H3_2pz']
vir_lmo1 =['H3_1s','H3_2s','H3_2pz']

vir_lmo2 =['H7_1s','H7_2s','H7_2px','H7_2py','H7_2pz']
vir_lmo2 =['H7_1s','H7_2s','H7_2pz']


df_F_C = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C.fc_ab = 0
for orb1 in vir_lmo:
    for orb2 in vir_lmo:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C.fc_ab += df.reset_index().fc_ab

#print(df_F_C)

df_F_C_1 = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C_1.fc_ab = 0
for orb1 in vir_lmo1:
    for orb2 in vir_lmo2:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_1.fc_ab += df.reset_index().fc_ab

df_F_C_2 = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C_2.fc_ab = 0
for orb1 in vir_lmo1:
    for orb2 in vir_lmo1:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_2.fc_ab += df.reset_index().fc_ab
#print(df_F_C_2)

df_F_C_3 = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C_3.fc_ab = 0
for orb1 in vir_lmo2:
    for orb2 in vir_lmo2:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_3.fc_ab += df.reset_index().fc_ab


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14,8))
ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
ax1.set_title(' a = [O-H1- O-H2], b = [O-H1- O-H2]')# f'a={orb1}, b={orb2}')

ax1.legend()

ax2.plot(ang, df_F_C_1.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, df_F_C_1.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
#ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
ax2.set_title('a = [O-H1], b = [O-H2]')# f'a={orb1}, b={orb2}')

ax2.legend()

ax3.plot(ang, df_F_C_2.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, df_F_C_2.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
#ax3.set_ylabel('Hz')
ax3.set_xlabel('Dihedral angle')
ax3.set_title('a = [O-H1], b = [O-H1]')# f'a={orb1}, b={orb2}')

ax3.legend()

ax4.plot(ang, df_F_C_3.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, df_F_C_3.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
#ax4.set_ylabel('Hz')
ax4.set_xlabel('Dihedral angle')
ax4.set_title('a = [O-H2], b = [O-H2]')# f'a={orb1}, b={orb2}')

ax4.legend()


plt.suptitle('FC contribution to $^3J(H_1-H_2)_{ia,jb}$ in H$_2$O$_2$ using 1s, 2s and 2p$_z$ antiligant')


plt.savefig(f'FC_C2H6_ab_pathways.png', dpi=200)


plt.show()               


