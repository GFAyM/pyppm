import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_pso_iajb_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'i', 'a', 'j', 'b']




#occ_lmo = ['F3_2s','F7_2s','F3_2pz','F7_2pz',
#'F3_2p2','F7_2p2','F3_2p1','F7_2p1']


occ_lmo = ['F3_2p1','F7_2p1','F3_2pz','F7_2pz','F3_2p2','F7_2p2']


lmo_vir = ["F3_2pz","F7_2pz","F3_3pz","F7_3pz","F3_3s","F7_3s","F3_3dz",
"F7_3dz","F3_3py","F7_3py","F3_3px","F7_3px","F3_3dxy","F7_3dxy","F3_3dx2-y2","F7_3dx2-y2"
,"F3_3dyz","F7_3dyz","F3_3dxz", "F7_3dxz", 'V_CC' ]

lmo_vir = ["F3_3py","F7_3py","F3_3s","F7_3s"]
#lmo_vir = ["F3_2pz","F7_2pz"]

lmo_vir1 = ["F3_2pz"]
lmo_vir2 = ["F7_2pz"]


df_F_C = data_J[(data_J.i.str.contains('F3_2p1') == True) & (data_J.j.str.contains('F3_2p1') == True) & 
(data_J.a.str.contains("F3_2pz") == True) & (data_J.b.str.contains("F3_2pz") == True)].reset_index()
df_F_C.fc_ab = 0
#for i in occ_lmo:
#    for j in occ_lmo:
for a in lmo_vir:
    for b in lmo_vir: 
        df_p1 = data_J[(data_J.i.str.contains('F3_2p1') == True) & (data_J.j.str.contains('F7_2p1') == True)
                & (data_J.a.str.contains(a) == True) & (data_J.b.str.contains(b) == True)].reset_index()

        df_p2 = data_J[(data_J.i.str.contains('F3_2p2') == True) & (data_J.j.str.contains('F7_2p2') == True)
                & (data_J.a.str.contains(a) == True) & (data_J.b.str.contains(b) == True)].reset_index()

        df_pz = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.j.str.contains('F7_2pz') == True)
                & (data_J.a.str.contains(a) == True) & (data_J.b.str.contains(b) == True)].reset_index()

#        if (df_p1.fc_ab + df_p2.fc_ab)[abs(df_p1.fc_ab + df_p2.fc_ab) > 0.5].any():
#        if (df_p1.fc_ab )[abs(df_p1.fc_ab) > 0.1].any():

        df_F_C.fc_ab += df_p1.fc_ab + df_p2.fc_ab + df_pz.fc_ab
#            df_F_C.fc_ab += df_p1.fc_ab 

        print(a,b)
#            print(df_F_C.fc_ab)
plt.figure(figsize=(10,8))
plt.plot(df_p1.ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J(F3-F7)_{ia,jb}$' )#f'a={orb1} b={orb2}')
#plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle(r' ^{PSO}(F3-F7)_{ia,jb}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
plt.title(r'i=F3-2p$_xyz$, a=F3(2p$_z$,3p$_{xy}$), j=F7-2p$_xyz$, b=F7(2p$_z$,3p$_{xy}$)')# f'a={orb1}, b={orb2}')
plt.savefig(f'FC_C2F2H4_iajb_F3F7.png', dpi=200)

plt.show()                #


