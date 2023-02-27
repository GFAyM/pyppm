import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_iajb_H2O2.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ij', 'fc_ab','i', 'a', 'j', 'b']

vir_lmo =['H1_1s', 'H2_1s', 'H1_2s', 'H2_2s', 'H1_2px', 'H2_2px','H1_2py', 'H2_2py', 'H1_2pz', 'H2_2pz']#, 
vir_lmo =['H1_1s', 'H2_1s', 'H1_2s', 'H2_2s', 'H1_2px', 'H2_2px']

occ_lmo = ['H1_lig', 'H2_lig']
occ_lmo1 = ['H1_lig']
occ_lmo2 = ['H2_lig']


vir_lmo1 = ['H1_1s','H1_2s', 'H1_2px','H1_2pz','H1_2py']#,'
vir_lmo2 = ['H2_1s','H2_2s', 'H2_2px','H2_2pz','H2_2py']#,


df_F_C = data_J[(data_J.a.str.contains('H1_1s') == True) & (data_J.b.str.contains('H1_1s') == True) 
                    & (data_J.i.str.contains('H1_lig') == True) & (data_J.j.str.contains('H2_lig') == True)].reset_index()
df_F_C.fc_ab = 0

for orb_i in occ_lmo:
    for orb_j in occ_lmo:
        for orb1_a in vir_lmo:
            for orb2_b in vir_lmo:
                df = data_J[(data_J.i.str.contains(orb_i) == True) &
                            (data_J.a.str.contains(orb1_a) == True) & 
                            (data_J.j.str.contains(orb_j) == True) &
                            (data_J.b.str.contains(orb2_b) == True) 
                             ]
                ang = df.ang
                df_F_C.fc_ab += df.reset_index().fc_ab


df_F_C_1 = data_J[(data_J.a.str.contains('H1_1s') == True) & (data_J.b.str.contains('H1_1s') == True) 
                    & (data_J.i.str.contains('H1_lig') == True) & (data_J.j.str.contains('H2_lig') == True)].reset_index()
df_F_C_1.fc_ab = 0
for orb_i in occ_lmo1:
    for orb_j in occ_lmo2:
        for orb1_a in vir_lmo1:
            for orb2_b in vir_lmo2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) &
                            (data_J.a.str.contains(orb1_a) == True) & 
                            (data_J.j.str.contains(orb_j) == True) &
                            (data_J.b.str.contains(orb2_b) == True) 
                             ]
                
                df_F_C_1.fc_ab += df.reset_index().fc_ab


df_F_C_2 = data_J[(data_J.a.str.contains('H1_1s') == True) & (data_J.b.str.contains('H1_1s') == True) 
                    & (data_J.i.str.contains('H1_lig') == True) & (data_J.j.str.contains('H2_lig') == True)].reset_index()
df_F_C_2.fc_ab = 0
for orb_i in occ_lmo1:
    for orb_j in occ_lmo1:
        for orb1_a in vir_lmo1:
            for orb2_b in vir_lmo1:
                df = data_J[(data_J.i.str.contains(orb_i) == True) &
                            (data_J.a.str.contains(orb1_a) == True) & 
                            (data_J.j.str.contains(orb_j) == True) &
                            (data_J.b.str.contains(orb2_b) == True) 
                             ]
                
                df_F_C_2.fc_ab += df.reset_index().fc_ab

df_F_C_3 = data_J[(data_J.a.str.contains('H1_1s') == True) & (data_J.b.str.contains('H1_1s') == True) 
                    & (data_J.i.str.contains('H1_lig') == True) & (data_J.j.str.contains('H2_lig') == True)].reset_index()
df_F_C_3.fc_ab = 0
for orb_i in occ_lmo2:
    for orb_j in occ_lmo2:
        for orb1_a in vir_lmo2:
            for orb2_b in vir_lmo2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) &
                            (data_J.a.str.contains(orb1_a) == True) & 
                            (data_J.j.str.contains(orb_j) == True) &
                            (data_J.b.str.contains(orb2_b) == True) 
                             ]
                
                df_F_C_3.fc_ab += df.reset_index().fc_ab



fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))
ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{FC}J(H-H)$')
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
ax1.set_title('i=j=[O-H1,O-H2], a=b=[O-H1*,O-H1*]')# f'a={orb1}, b={orb2}')

ax1.legend()

ax2.plot(ang, df_F_C_1.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, df_F_C_1.fc_ij, 'g<-', label='$^{FC}J(H-H)$')
ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
ax2.set_title('i=O-H1, j=O-H2, a=O-H1*, b=O-H2*')# f'a={orb1}, b={orb2}')

ax2.legend()

ax3.plot(ang, df_F_C_2.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, df_F_C_2.fc_ij, 'g<-', label='$^{FC}J(H-H)$')
ax3.set_ylabel('Hz')
ax3.set_xlabel('Dihedral angle')
ax3.set_title('i=j=O-H1, a=b=O-H1*')# f'a={orb1}, b={orb2}')

ax3.legend()

ax4.plot(ang, df_F_C_3.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, df_F_C_3.fc_ij, 'g<-', label='$^{FC}J(H-H)$')
ax4.set_ylabel('Hz')
ax4.set_xlabel('Dihedral angle')
ax4.set_title('i=j=O-H2, a=b=O-H2*')# f'a={orb1}, b={orb2}')

ax4.legend()


plt.suptitle('FC contribution to $^3J(H_1-H_2)_{ia,jb}$ in H$_2$O$_2$')


plt.savefig(f'FC_occ_H2O2_iajb_pathways.png', dpi=200)


plt.show()               


