import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_iajb_C2H6_ccpvdz.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'i', 'a', 'j', 'b', 'fc_ij']

occ_lmo = ['lig_1','lig_2']
occ_lmo1 = ['lig_1']
occ_lmo2 = ['lig_2']

vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s', 'H3_2px', 'H7_2px','H3_2py', 'H7_2py', 'H3_2pz', 'H7_2pz']
vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s']

# 'O1_3dz', 'O2_3dz', , 'O1_3py', 'O2_3py', ,'O1_3ddxz','O2_3ddxz']#, 
# 'O1_3dx', 'O2_3dx','O1_3dy', 'O2_3dy','O1_3s', 'O2_3s'
#vir_lmo = ['H3_2s', 'H4_2s', 'H3_2px', 'H4_2px','H3_2py', 'H4_2py', 'H3_2pz', 'H4_2pz','O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']

#'O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']

vir_lmo1 =['H3_1s','H3_2s','H3_2px','H3_2py','H3_2pz']
vir_lmo1 =['H3_1s','H3_2s']

vir_lmo2 =['H7_1s','H7_2s','H7_2px','H7_2py','H7_2pz']
vir_lmo2 =['H7_1s','H7_2s']


df_F_C = data_J[(data_J.i.str.contains('lig_1') == True) & (data_J.j.str.contains('lig_1') == True) &
            (data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C.fc_ab = 0

for orb_i in occ_lmo:
    for orb_j in occ_lmo:
        for orb_a in vir_lmo:
            for orb_b in vir_lmo:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C.fc_ab += df.reset_index().fc_ab


df_F_C_1 = data_J[(data_J.i.str.contains('lig_1') == True) & (data_J.j.str.contains('lig_1') == True) &
            (data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C_1.fc_ab = 0

for orb_i in occ_lmo1:
    for orb_j in occ_lmo2:
        for orb_a in vir_lmo1:
            for orb_b in vir_lmo2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C_1.fc_ab += df.reset_index().fc_ab

df_F_C_2 = data_J[(data_J.i.str.contains('lig_1') == True) & (data_J.j.str.contains('lig_1') == True) &
            (data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C_2.fc_ab = 0

for orb_i in occ_lmo1:
    for orb_j in occ_lmo1:
        for orb_a in vir_lmo1:
            for orb_b in vir_lmo1:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C_2.fc_ab += df.reset_index().fc_ab

df_F_C_3 = data_J[(data_J.i.str.contains('lig_1') == True) & (data_J.j.str.contains('lig_1') == True) &
            (data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C_3.fc_ab = 0

for orb_i in occ_lmo2:
    for orb_j in occ_lmo2:
        for orb_a in vir_lmo2:
            for orb_b in vir_lmo2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C_3.fc_ab += df.reset_index().fc_ab


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))
ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
ax1.set_title('i=j=[C-H1;C-H2], a=b=[O-H1*;O-H2*]')# f'a={orb1}, b={orb2}')

ax1.legend()

ax2.plot(ang, df_F_C_1.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, df_F_C_1.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
ax2.set_title('i=C-H1, a=C-H1*, j=C-H2, b=C-H2*')# f'a={orb1}, b={orb2}')

ax2.legend()

ax3.plot(ang, df_F_C_2.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, df_F_C_2.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
ax3.set_ylabel('Hz')
ax3.set_xlabel('Dihedral angle')
ax3.set_title('i=j=C-H1, a=b=C-H1*')# f'a={orb1}, b={orb2}')

ax3.legend()

ax4.plot(ang, df_F_C_3.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, df_F_C_3.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
ax4.set_ylabel('Hz')
ax4.set_xlabel('Dihedral angle')
ax4.set_title('i=j=C-H1, a=b=C-H1*')# f'a={orb1}, b={orb2}')

ax4.legend()


plt.suptitle('FC contribution to $^3J(H_1-H_2)_{ia,jb}$ in H$_2$C$_6$, using antiligants 1s y 2s.')


plt.savefig(f'FC_iajb_C2H6.png', dpi=200)


plt.show()               


