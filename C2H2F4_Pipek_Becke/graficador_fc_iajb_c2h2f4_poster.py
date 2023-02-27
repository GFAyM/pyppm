import matplotlib.pyplot as plt
import pandas as pd

text = 'pathways_fc_iajb_c2h2f4.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fc_tot', 'fcsd', 'a', 'b']


lmo_vir1 = ["H3_2pz","H3_1s","H3_2s","H3_2px","H3_2py","C1_2pz"]
lmo_vir2 = ["H7_2pz","H7_1s","H7_2s","H7_2px","H7_2py","C2_2pz"]
lmo_occ = ['CH1', 'CH2']

df_F_C_1 = data_J[ (data_J.a.str.contains('H1_2pz') == True)
                &  (data_J.b.str.contains('H2_2pz') == True)].reset_index()

df_F_C_2 = data_J[ (data_J.a.str.contains('C-H_1') == True)
                &  (data_J.b.str.contains('H2_1s') == True)].reset_index()

df_F_C_3 = data_J[ (data_J.a.str.contains('C-H_1') == True)
                &  (data_J.b.str.contains('C-H_2') == True)].reset_index()

df_F_C_4 = data_J[ (data_J.a.str.contains('H1_2s') == True)
                &  (data_J.b.str.contains('H2_2s') == True)].reset_index()


#print(df.fcsd)
#print(df.reset_index().fcsd)
#df_F_C.fcsd += df.reset_index().fcsd
#b = df.b
#fc = df.fc
#fc_ab = df.fc_ab
plt.figure(figsize=(10,12))
plt.plot(df_F_C_1.ang, df_F_C_1.fcsd, 'b>-', label=r'i=C-H$_1$ a=C-H1*, i=C-H$_2$ a=C-H*')
plt.plot(df_F_C_2.ang, df_F_C_2.fcsd, 'g>-', label=r'i=C-H$_1$ a=2p$_z$, i=C-H$_2$ a=2p$_z$')
plt.plot(df_F_C_3.ang, df_F_C_3.fcsd, 'r>-', label=r'i=C-H$_1$ a=C-H1**, i=C-H$_2$ a=C-H2**')
plt.plot(df_F_C_4.ang, df_F_C_4.fcsd, 'c-.', label=r'i=C-H$_1$ a=2s, i=LP$_2$ a=2s' )#f'a={orb1} b={orb2}')
#plt.plot(data_J_tot.ang, data_J_tot.fc, 'm-.', label=r'J_{ij}' )

#plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('FC+SD contribution to $^3J(H-H)_{ia,jb}$ in C$_2$F$_2$H$_4$ with cc-pVDZ basis')
#plt.title('i=C-F$_1$(2s,2p$_z$, 2p$_x$), j=C-F$_2$(2s,2p$_z$,2p$_x$), a = b = all')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FCSD_pathways.png', dpi=200)
plt.show()                #


