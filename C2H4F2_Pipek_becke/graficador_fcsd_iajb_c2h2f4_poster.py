import matplotlib.pyplot as plt
import pandas as pd

text = 'cloppa_fcsd_iajb_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fcsd', 'i', 'a', 'j', 'b']

data_J_tot = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
data_J_tot.columns = ['ang', 'fcsd', 'fc', 'pso']

occ_lmo1 = ['F3_2pz','F3_2s']
occ_lmo2 = ['F7_2pz','F7_2s']

lmo_vir1 = ["F3_2pz_","F3_3pz_","F3_3s_", "F3_3py_", "F3_3px_"]

lmo_vir2 = ["F7_2pz_","F7_3pz_","F7_3s_", "F7_3py_", "F7_3px_"]

df_F_C_1 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()

df_F_C_2 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()

df_F_C_3 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.a.str.contains('F3_3pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()

df_F_C_4 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()

df_F_C_5 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.a.str.contains('F3_3pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()

df_F_C_6 = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.a.str.contains('F3_3pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()

df_F_C_7 = data_J[(data_J.i.str.contains('F3_2s') == True) & (data_J.a.str.contains('F3_3pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()

df_F_C_8 = data_J[(data_J.i.str.contains('F3_2s') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()
df_F_C_9 = data_J[(data_J.i.str.contains('F3_2s') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()


#print(df.fcsd)
#print(df.reset_index().fcsd)
#df_F_C.fcsd += df.reset_index().fcsd
#b = df.b
#fc = df.fc
#fc_ab = df.fc_ab
plt.figure(figsize=(10,8))
plt.plot(df_F_C_1.ang, df_F_C_1.fcsd, 'b>-', label=r'i=C-F$_1$ a=F2$_1$p$_z$, i=C-H$_2$ a=F$_2$2p$_z$')
plt.plot(df_F_C_2.ang, df_F_C_2.fcsd, 'g>-', label=r'i=C-F$_1$ a=F2$_1$p$_z$, i=C-H$_2$ a=F$_2$3p$_z$')
plt.plot(df_F_C_3.ang, df_F_C_3.fcsd, 'r>-', label=r'i=C-F$_1$ a=F3$_1$p$_z$, i=C-H$_2$ a=F$_2$2p$_z$')
plt.plot(df_F_C_4.ang, df_F_C_4.fcsd, 'c-.', label=r'i=C-F$_1$ a=F2$_1$p$_z$, i=LP$_2$ a=F$_2$2p$_z$' )#f'a={orb1} b={orb2}')
plt.plot(df_F_C_5.ang, df_F_C_5.fcsd, 'm-.', label=r'i=C-F$_1$ a=F3$_1$p$_z$, i=LP$_2$ a=F$_2$2p$_z$' )
plt.plot(df_F_C_6.ang, df_F_C_6.fcsd, 'b-.', label=r'i=C-F$_1$ a=F3$_1$p$_z$, i=LP$_2$ a=F$_2$3p$_z$' )
plt.plot(df_F_C_7.ang, df_F_C_7.fcsd, 'g:', label=r'i=LP$_1$ a=F$_1$3p$_z$, i=LP$_2$ a=F$_2$3p$_z$' )
plt.plot(df_F_C_8.ang, df_F_C_8.fcsd, 'r:', label=r'i=LP$_1$ a=F$_1$2p$_z$, i=LP$_2$ a=F$_2$3p$_z$')
plt.plot(df_F_C_9.ang, df_F_C_9.fcsd, 'c:', label=r'i=LP$_1$ a=F$_1$2p$_z$, i=LP$_2$ a=F$_2$2p$_z$')

#plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Dihedral Angle')
plt.title('FC+SD contribution to $^3J$(F-F)$_{ia,jb}$ in C$_2$H$_4$F$_2$ using Pipek-Mezey LMO with cc-pVDZ basis')
#plt.title('i=C-F$_1$(2s,2p$_z$, 2p$_x$), j=C-F$_2$(2s,2p$_z$,2p$_x$), a = b = all')# f'a={orb1}, b={orb2}')
plt.savefig(f'FCSD_pathways.png', dpi=400)
plt.show()                #


