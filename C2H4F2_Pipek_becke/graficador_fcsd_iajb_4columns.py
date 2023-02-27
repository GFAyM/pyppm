import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fcsd_iajb_C2H4F2_2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fcsd', 'i', 'a', 'j', 'b']

occ_lmo1 = ['F3_2pz','F3_2s','F3_LPx','F3_LPy']
occ_lmo2 = ['F7_2pz','F7_2s','F7_LPx','F7_LPy']

lmo_vir1 = ["F3_2pz_","F3_3pz_","F3_3s_", "F3_3py_", "F3_3px_"]
#lmo_vir1 = ["F3_2pz_","F3_3pz_","F3_3s_"]

lmo_vir2 = ["F7_2pz_","F7_3pz_","F7_3s_", "F7_3py_", "F7_3px_"]
#lmo_vir2 = ["F7_2pz_","F7_3pz_","F7_3s_"]


df_F_C = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.j.str.contains('F7_2pz') == True) &
            (data_J.a.str.contains('F3_2pz_') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
df_F_C
df_F_C.fcsd = 0

for orb_i in occ_lmo1:
    for orb_j in occ_lmo2:
        for orb_a in lmo_vir1:
            for orb_b in lmo_vir2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C.fcsd += df.reset_index().fcsd

df_F_C_1 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.j.str.contains('F7_2pz') == True) &
            (data_J.a.str.contains('F3_2pz_') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()

df_F_C_1.fcsd = 0

lmo_vir1 = ["F3_2pz_","F3_3pz_","F3_3s_"]
lmo_vir2 = ["F7_2pz_","F7_3pz_","F7_3s_"]

for orb_i in occ_lmo1:
    for orb_j in occ_lmo2:
        for orb_a in lmo_vir1:
            for orb_b in lmo_vir2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C_1.fcsd += df.reset_index().fcsd

lmo_vir1 = ["F3_2pz_","F3_3pz_"]
lmo_vir2 = ["F7_2pz_","F7_3pz_"]


df_F_C_2 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.j.str.contains('F7_2pz') == True) &
            (data_J.a.str.contains('F3_2pz_') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
df_F_C_2
df_F_C_2.fcsd = 0


for orb_i in occ_lmo1:
    for orb_j in occ_lmo2:
        for orb_a in lmo_vir1:
            for orb_b in lmo_vir2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C_2.fcsd += df.reset_index().fcsd

lmo_vir1 = ["F3_3pz_"]
lmo_vir2 = ["F7_3pz_"]


df_F_C_5 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.j.str.contains('F7_2pz') == True) &
            (data_J.a.str.contains('F3_2pz_') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
df_F_C_5
df_F_C_5.fcsd = 0


for orb_i in occ_lmo1:
    for orb_j in occ_lmo2:
        for orb_a in lmo_vir1:
            for orb_b in lmo_vir2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C_5.fcsd += df.reset_index().fcsd



text = 'cloppa_fcsd_ij_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fcsd', 'a', 'b']
df_F_C_3 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_3.fcsd = 0
for orb1 in occ_lmo1:
    for orb2 in occ_lmo2:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_3.fcsd += df.reset_index().fcsd


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,10))
ax1.plot(ang, df_F_C.fcsd, 'b>-', label='$^{FC+SD}J_{ia,jb}(F-F)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, df_F_C_3.fcsd, 'g<-', label='$^{FC+SD}J_{ij}(F-F)$')
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
ax1.set_title('virt = [2p$_z$, 3p$_z$, 3s, 3p$_{xy}$]')# f'a={orb1}, b={orb2}')

ax1.legend()

ax2.plot(ang, df_F_C_1.fcsd, 'b>-', label='$^{FC+SD}J_{ia,jb}(F-F)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, df_F_C_3.fcsd, 'g<-', label='$^{FC+SD}J_{ij}(F-F)$')

ax2.set_xlabel('Dihedral angle')
ax2.set_title('virt = [2p$_z$, 3p$_z$, 3s]')# f'a={orb1}, b={orb2}')

ax2.legend()

ax3.plot(ang, df_F_C_2.fcsd, 'b>-', label='$^{FC+SD}J_{ia,jb}(F-F)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, df_F_C_3.fcsd, 'g<-', label='$^{FC+SD}J_{ij}(F-F)$')
ax3.set_xlabel('Dihedral angle')
ax3.set_title('virt = [2p$_z$, 3p$_z$]')# f'a={orb1}, b={orb2}')

ax3.legend()

ax4.plot(ang, df_F_C_5.fcsd, 'b>-', label='$^{FC+SD}J_{ia,jb}(F-F)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, df_F_C_3.fcsd, 'g<-', label='$^{FC+SD}J_{ij}(F-F)$')

ax4.set_xlabel('Dihedral angle')
ax4.set_title('virt = [3p$_z$]')# f'a={orb1}, b={orb2}')

ax4.legend()


plt.suptitle('FC+SD contribution of $^3J(F_1-F_2)_{ia,jb}$ to $^3J(F_1-F_2)_{i,j}$ in H$_2$C$_4$F$_2$.')


plt.savefig(f'FCSD_iajb_C2H4F2.png', dpi=200)


plt.show()               


