import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'pathways_fc_iajb_c2h2f4.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang','fc_ij', 'fc_iajb', 'a', 'b']

lmo_vir1 = [ 'C-H_1']

lmo_vir2 = ['C-H_2']

df_F_C = data_J[(data_J.a.str.contains('H1_2pz') == True) & (data_J.b.str.contains('H2_2pz') == True)].reset_index()
df_F_C.fc_iajb = 0

#print(df_F_C)
for orb1 in lmo_vir1:
    for orb2 in lmo_vir2:

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C.fc_iajb += df.reset_index().fc_iajb

        

df_F_C_1 = data_J[(data_J.a.str.contains('H1_2pz') == True) & (data_J.b.str.contains('H2_2pz') == True)].reset_index()
df_F_C_1.fc_iajb = 0

lmo_vir1 = ['C-H_1', 'H1_2pz']

lmo_vir2 = ['C-H_2', 'H2_2pz']

for orb1 in lmo_vir1:
    for orb2 in lmo_vir2:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_1.fc_iajb += df.reset_index().fc_iajb
        
df_F_C_2 = data_J[(data_J.a.str.contains('H1_2pz') == True) & (data_J.b.str.contains('H2_2pz') == True)].reset_index()
df_F_C_2.fc_iajb = 0

lmo_vir1 = ['C-H_1', 'H1_2pz','H1_1s']

lmo_vir2 = ['C-H_2', 'H2_2pz','H2_1s']
for orb1 in lmo_vir1:
    for orb2 in lmo_vir2:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_2.fc_iajb += df.reset_index().fc_iajb


lmo_vir1 = [ 'C-H_1', 'H1_2pz', 'H1_1s', 'H1_2s']#, 'H1_2px', 'H1_2py']

lmo_vir2 = ['C-H_2', 'H2_2pz', 'H2_1s', 'H2_2s']#, 'H2_2px', 'H2_2py' ]
df_F_C_3 = data_J[(data_J.a.str.contains('H1_2pz') == True) & (data_J.b.str.contains('H2_2pz') == True)].reset_index()
df_F_C_3.fc_iajb = 0
for orb1 in lmo_vir1:
    for orb2 in lmo_vir2:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_3.fc_iajb += df.reset_index().fc_iajb


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))

ax1.plot(ang, df_F_C.fc_iajb, 'b>-', label='$^{fc}J_{iajb}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{fc}J_{jb}(H-H)$')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
plt.suptitle('AntiLigants contributions to $^{FC}J(H-H)_{i,j}$ in C$_2$F$_4$H$_2$')
ax1.set_title(r'a=C-H1*, b=C-H2*')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)


#ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
ax2.plot(ang, df_F_C_1.fc_iajb, 'b>-', label='$^{FC}J_{iajb}(H-H)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{FC}_{ij}J(H-H)$')
ax2.legend()
ax2.set_title(r'a=C-H1*, b=C-H2*')# f'a={orb1}, b={orb2}')

#ax3.set_ylabel('Hz')
ax3.set_xlabel('Dihedral angle')
ax3.plot(ang, df_F_C_2.fc_iajb, 'b>-', label='$^{FC}J_{iajb}(H-H)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{FC}_{ij}J(H-H)$')
ax3.legend()
ax3.set_title(r'a=C-H1*, b=C-H2*')# f'a={orb1}, b={orb2}')

ax4.set_xlabel('Dihedral angle')
ax4.plot(ang, df_F_C_3.fc_iajb, 'b>-', label='$^{FC}J_{iajb}(H-H)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{FC}_{ij}J(H-H)$')
ax4.legend()
ax4.set_title(r'a=C-H1*, b=C-H2*')# f'a={orb1}, b={orb2}')

plt.savefig('FC_iajb_C2H2F4_ccpvdz.png')
plt.show()    

