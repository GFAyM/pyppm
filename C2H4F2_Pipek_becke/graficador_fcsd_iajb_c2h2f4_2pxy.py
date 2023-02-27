import matplotlib.pyplot as plt
import pandas as pd

text = 'cloppa_fcsd_iajb_C2H4F2_2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fcsd', 'i', 'a', 'j', 'b']





df_F_C_10 = data_J[(data_J.i.str.contains('F3_LPx') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
df_F_C_11 = data_J[(data_J.i.str.contains('F3_LPx') == True) & (data_J.a.str.contains('F3_3pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()
df_F_C_12 = data_J[(data_J.i.str.contains('F3_LPx') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()
df_F_C_13 = data_J[(data_J.i.str.contains('F3_LPx') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
df_F_C_14 = data_J[(data_J.i.str.contains('F3_LPx') == True) & (data_J.a.str.contains('F3_3pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()
df_F_C_15 = data_J[(data_J.i.str.contains('F3_LPx') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()

df_F_C_1 = data_J[(data_J.i.str.contains('F3_LPy') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
df_F_C_2 = data_J[(data_J.i.str.contains('F3_LPy') == True) & (data_J.a.str.contains('F3_3pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()
df_F_C_3 = data_J[(data_J.i.str.contains('F3_LPy') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()
df_F_C_4 = data_J[(data_J.i.str.contains('F3_LPy') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
df_F_C_5 = data_J[(data_J.i.str.contains('F3_LPy') == True) & (data_J.a.str.contains('F3_3pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()
df_F_C_6 = data_J[(data_J.i.str.contains('F3_LPy') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2s') == True) & (data_J.b.str.contains('F7_3pz_') == True)].reset_index()


plt.figure(figsize=(12,10))
plt.plot(df_F_C_1.ang, df_F_C_1.fcsd, 'bo--', label=r'i=LP$_1$y a=F$_1$2p$_z$, j=C-H$_2$ b=F$_2$2p$_z$')
plt.plot(df_F_C_2.ang, df_F_C_2.fcsd, 'go--', label=r'i=LP$_1$y a=F$_1$3p$_z$, j=C-H$_2$ b=F$_2$3p$_z$')
plt.plot(df_F_C_3.ang, df_F_C_3.fcsd, 'ro--', label=r'i=LP$_1$y a=F$_1$3p$_z$, j=C-H$_2$ b=F$_2$3p$_z$')
plt.plot(df_F_C_4.ang, df_F_C_4.fcsd, 'cv--', label=r'i=LP$_1$y a=F$_1$2p$_z$, j=LP$_2$ b=F$_2$2p$_z$' )
plt.plot(df_F_C_5.ang, df_F_C_5.fcsd, 'mv--', label=r'i=LP$_1$y a=F$_2$3p$_z$, j=LP$_2$ b=F$_2$2p$_z$' )
plt.plot(df_F_C_6.ang, df_F_C_6.fcsd, 'bv--', label=r'i=LP$_1$y a=F$_1$3p$_z$, j=LP$_2$ b=F$_2$3p$_z$' )

plt.plot(df_F_C_10.ang, df_F_C_10.fcsd, 'b<--', label=r'i=LP$_1$x a=F$_1$3p$_z$, j=C-H$_2$ b=F$_2$3p$_z$' )
plt.plot(df_F_C_11.ang, df_F_C_11.fcsd, 'r<--', label=r'i=LP$_1$x a=F$_1$2p$_z$, j=C-H$_2$ b=F$_2$3p$_z$')
plt.plot(df_F_C_12.ang, df_F_C_12.fcsd, 'c<--', label=r'i=LP$_1$x a=F$_1$2p$_z$, j=C-H$_2$ b=F$_2$2p$_z$')
#plt.plot(df_F_C_13.ang, df_F_C_13.fcsd, 'b*--', label=r'i=LP$_1$x a=F$_1$3p$_z$, i=LP$_2$ a=F$_2$3p$_z$' )
#plt.plot(df_F_C_14.ang, df_F_C_14.fcsd, 'r*--', label=r'i=LP$_1$x a=F$_1$2p$_z$, i=LP$_2$ a=F$_2$3p$_z$')
#plt.plot(df_F_C_15.ang, df_F_C_15.fcsd, 'c*--', label=r'i=LP$_1$x a=F$_1$2p$_z$, i=LP$_2$ a=F$_2$2p$_z$')

#plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Dihedral Angle')
plt.title('FC+SD contribution to $^3J$(F-F)$_{ia,jb}$ in C$_2$H$_4$F$_2$ using Pipek-Mezey LMO with cc-pVDZ basis')
#plt.title('i=C-F$_1$(2s,2p$_z$, 2p$_x$), j=C-F$_2$(2s,2p$_z$,2p$_x$), a = b = all')# f'a={orb1}, b={orb2}')
plt.savefig(f'FCSD_pathways_pxy.png', dpi=400)
plt.show()                #


