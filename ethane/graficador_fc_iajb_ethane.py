import matplotlib.pyplot as plt
import pandas as pd

text = 'cloppa_fc_vir_C2H6_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']


vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s', 'H3_2px', 'H7_2px','H3_2py', 'H7_2py', 'H3_2pz', 'H7_2pz']




lmo_vir1 = ['H1_1s','H1_2s', 'H1_2px', 'H1_2py', 'H1_2pz']

lmo_vir2 = ['H2_1s','H2_2s', 'H2_2px', 'H2_2py', 'H2_2pz']

df_F_C_1 = data_J[ (data_J.a.str.contains('H3_1s') == True)
                &  (data_J.b.str.contains('H7_1s') == True)].reset_index()

df_F_C_2 = data_J[ (data_J.a.str.contains('H3_2s') == True)
                &  (data_J.b.str.contains('H7_2s') == True)].reset_index()
df_F_C_3 = data_J[ (data_J.a.str.contains('H3_2pz') == True)
                &  (data_J.b.str.contains('H7_2pz') == True)].reset_index()
df_F_C_4 = data_J[ (data_J.a.str.contains('H3_2px') == True)
                &  (data_J.b.str.contains('H7_2px') == True)].reset_index()
df_F_C_5 = data_J[ (data_J.a.str.contains('H3_2py') == True)
                &  (data_J.b.str.contains('H7_2py') == True)].reset_index()



#print(df.fcsd)
#print(df.reset_index().fcsd)
#df_F_C.fcsd += df.reset_index().fcsd
#b = df.b
#fc = df.fc
#fc_ab = df.fc_ab
plt.figure(figsize=(10,8))
#plt.plot(df_F_C_1.ang, df_F_C_1.fc_ab, 'b>-', label=r'i=C-F$_1$ a=F2$_1$p$_z$, i=C-H$_2$ a=F$_2$2p$_z$')
#plt.plot(df_F_C_2.ang, df_F_C_2.fc_ab, 'g>-', label=r'i=C-F$_1$ a=F2$_1$p$_z$, i=C-H$_2$ a=F$_2$3p$_z$')
plt.plot(df_F_C_3.ang, df_F_C_3.fc_ab, 'r>-', label=r'i=C-F$_1$ a=F3$_1$p$_z$, i=C-H$_2$ a=F$_2$2p$_z$')
plt.plot(df_F_C_4.ang, df_F_C_4.fc_ab, 'c-.', label=r'i=C-F$_1$ a=F2$_1$p$_z$, i=LP$_2$ a=F$_2$2p$_z$' )#f'a={orb1} b={orb2}')
plt.plot(df_F_C_5.ang, df_F_C_5.fc_ab, 'm-.', label=r'i=C-F$_1$ a=F3$_1$p$_z$, i=LP$_2$ a=F$_2$2p$_z$' )
#plt.plot(df_F_C_6.ang, df_F_C_6.fcsd, 'b-.', label=r'i=C-F$_1$ a=F3$_1$p$_z$, i=LP$_2$ a=F$_2$3p$_z$' )
#plt.plot(df_F_C_7.ang, df_F_C_7.fcsd, 'g:', label=r'i=LP$_1$ a=F$_1$3p$_z$, i=LP$_2$ a=F$_2$3p$_z$' )
#plt.plot(df_F_C_8.ang, df_F_C_8.fcsd, 'r:', label=r'i=LP$_1$ a=F$_1$2p$_z$, i=LP$_2$ a=F$_2$3p$_z$')
#plt.plot(df_F_C_9.ang, df_F_C_9.fcsd, 'c:', label=r'i=LP$_1$ a=F$_1$2p$_z$, i=LP$_2$ a=F$_2$2p$_z$')

#plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Dihedral Angle')
plt.title('FC contribution to $^3J$(F-F)$_{ia,jb}$ in C$_2$H$_6$ using Pipek-Mezey LMO with cc-pVDZ basis')
#plt.title('i=C-F$_1$(2s,2p$_z$, 2p$_x$), j=C-F$_2$(2s,2p$_z$,2p$_x$), a = b = all')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FCSD_pathways_H2O2.png', dpi=400)
plt.show()                #


