import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fcsd_ij_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fcsd', 'a', 'b']

occ_lmo = ['F3_2pz','F7_2pz']#,'F3_2s', 'F7_2s','F3_LPx','F7_LPx','F3_LPy','F7_LPy'
occ_lmo = ['F3_2s', 'F7_2s']
occ_lmo = ['F3_LPx','F7_LPx','F3_LPy','F7_LPy']
occ_lmo = ['F3_LPx','F7_LPx','F3_LPy','F7_LPy','F3_2s', 'F7_2s']
#occ_lmo = ['F3_2pz','F7_2pz','F3_2s', 'F7_2s','F3_LPx','F7_LPx','F3_LPy','F7_LPy']
occ_lmo = ['F3_2s', 'F7_2s','F3_2pz','F7_2pz']


occ_lmo_1 = ['F3_2pz'] #,'F3_2s''F3_LPx','F3_LPy',,
occ_lmo_1 = ['F3_2s'] #,'F3_2s''F3_LPx','F3_LPy',,
occ_lmo_1 = ['F3_LPx','F3_LPy'] #,'F3_2s',,
occ_lmo_1 = ['F3_2s', 'F3_LPx','F3_LPy'] #,'F3_2s',,
occ_lmo_1 = ['F3_2pz','F3_2s','F3_LPx','F3_LPy']
occ_lmo_1 = ['F3_2s','F3_2pz']


occ_lmo_2 = ['F7_2pz'] #, 'F7_LPx','F7_LPy',
occ_lmo_2 = ['F7_2s'] 
occ_lmo_2 = ['F7_LPx','F7_LPy'] 
occ_lmo_2 = ['F7_LPx','F7_LPy','F7_2s'] 
occ_lmo_2 = ['F7_LPx','F7_LPy'] 
occ_lmo_2 = ['F7_2s','F7_2pz']

df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C.fcsd = 0

for orb1 in occ_lmo:
    for orb2 in occ_lmo:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C.fcsd += df.reset_index().fcsd
        
df_F_C_1 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_1.fcsd = 0
for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_2:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_1.fcsd += df.reset_index().fcsd
        
df_F_C_2 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_2.fcsd = 0
for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_1:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_2.fcsd += df.reset_index().fcsd

df_F_C_3 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_3.fcsd = 0
for orb1 in occ_lmo_2:
    for orb2 in occ_lmo_2:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        df_F_C_3.fcsd += df.reset_index().fcsd


data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
data_J = pd.DataFrame(data_J)
fcsd = data_J[1]
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))

ax1.plot(ang, df_F_C.fcsd, 'b>-', label='$^{FCSD}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, fcsd, 'g<-', label='$^{FCSD}J(F-F)$')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
plt.suptitle('FC+SD contribution to $^3J(F-F)_{i,j}$ in C$_2$F$_2$H$_4$ with Ligants and Free Pairs 2s and 2p$_{xy}$ LMOs')
ax1.set_title(r'i=j=[C-F1, C-F2, PL1, PL2]')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)


#ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
ax2.plot(ang, df_F_C_1.fcsd, 'b>-', label='$^{FCSD}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, fcsd, 'g<-', label='$^{FCSD}J(F-F)$')
ax2.legend()
ax2.set_title(r'i=[C-F1,PL1], j=[C-F2, PL2]')# f'a={orb1}, b={orb2}')

#ax3.set_ylabel('Hz')
ax3.set_xlabel('Dihedral angle')
ax3.plot(ang, df_F_C_2.fcsd, 'b>-', label='$^{FCSD}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, fcsd, 'g<-', label='$^{FCSD}J(F-F)$')
ax3.legend()
ax3.set_title(r'i=j=[C-F1, PL1]')# f'a={orb1}, b={orb2}')

ax4.set_xlabel('Dihedral angle')
ax4.plot(ang, df_F_C_3.fcsd, 'b>-', label='$^{FCSD}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, fcsd, 'g<-', label='$^{FCSD}J(F-F)$')
ax4.legend()
ax4.set_title(r'i=j=[C-F2,PL2]')# f'a={orb1}, b={orb2}')

#plt.savefig('C2F2H4_fcsd_occ_lig_PL_2s_2pxy.png')
plt.show()    

