import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_pso_ij_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso', 'a', 'b']

occ_lmo = ['F3_2pz','F7_2pz','F3_LPx','F7_LPx','F3_LPy','F7_LPy','F3_2s', 'F7_2s']
#occ_lmo = ['F3_2pz','F7_2pz','F3_LPx','F7_LPx','F3_LPy','F7_LPy']

occ_lmo_1 = ['F3_2pz','F3_LPx','F3_LPy','F3_2s']
#occ_lmo_1 = ['F3_2pz','F3_LPx','F3_LPy']

occ_lmo_2 = ['F7_2pz','F7_LPx','F7_LPy','F7_2s']
#occ_lmo_2 = ['F7_2pz','F7_LPx','F7_LPy']

df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C.pso = 0

for orb1 in occ_lmo:
    for orb2 in occ_lmo:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.pso[abs(df.pso) > 0].any():
            ang = df.ang
            df_F_C.pso += df.reset_index().pso


df_F_C_1 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_1.pso = 0

for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_1:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.pso[abs(df.pso) > 0].any():
            ang = df.ang
            df_F_C_1.pso += df.reset_index().pso
            
df_F_C_2 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_2.pso = 0
for orb1 in occ_lmo_2:
    for orb2 in occ_lmo_2:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.pso[abs(df.pso) > 0].any():
            ang = df.ang
            df_F_C_2.pso += df.reset_index().pso
            
df_F_C_3 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_3.pso = 0
for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_2:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.pso[abs(df.pso) > 0].any():
            ang = df.ang
            df_F_C_3.pso += df.reset_index().pso
            
data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
data_J = pd.DataFrame(data_J)
PSO = data_J[3]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,10))

ax1.plot(ang, df_F_C.pso, 'b>-', label='$^{PSO}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, PSO, 'g<-', label='$^{PSO}J(F-F)$')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
plt.suptitle('PSO contribution to $^3J(F-F)_{i,j}$ in C$_2$F$_2$H$_4$ with 2p occupied LMOs')
ax1.set_title(r'i=[F$_1$2p$_{zxy},2s$, F$_2$2p$_{zxy},2s$]')# f'a={orb1}, b={orb2}')



ax2.plot(ang, df_F_C_1.pso, 'b>-', label='$^{PSO}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax2.plot(ang, PSO, 'g<-', label='$^{PSO}J(F-F)$')

ax2.legend()
ax2.set_ylabel('Hz')
ax2.set_xlabel('Dihedral angle')
plt.suptitle('PSO contribution to $J(F-F)_{i,j}$ in C$_2$F$_2$H$_4$ with 2p$_{zxy}$ occupied LMOs and lone pair 2s')
ax2.set_title(r'i=F$_1$2p$_{zxy},2s$, j=F$_1$2p$_{zxy},2s$')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)


#ax2.set_ylabel('Hz')
ax3.set_xlabel('Dihedral angle')
ax3.plot(ang, df_F_C_2.pso, 'b>-', label='$^{PSO}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax3.plot(ang, PSO, 'g<-', label='$^{PSO}J(F-F)$')
ax3.legend()
ax3.set_title(r'i=F$_2$2p$_{zxy},2s$, j=F$_2$2p$_{zxy},2s$')# f'a={orb1}, b={orb2}')

#ax3.set_ylabel('Hz')
ax4.set_xlabel('Dihedral angle')
ax4.plot(ang, df_F_C_3.pso, 'b>-', label='$^{PSO}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
ax4.plot(ang, PSO, 'g<-', label='$^{PSO}J(F-F)$')
ax4.legend(loc=8)
ax4.set_title(r'i=F$_1$2p$_{zxy}$,2s, j=F$_2$2p$_{zxy},2s$')# f'a={orb1}, b={orb2}')

plt.savefig('C2F2H4_pso_ij_2pzxy_2s.png')
plt.show()    

