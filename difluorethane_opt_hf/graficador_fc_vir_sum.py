import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_iajb_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'i', 'a', 'j', 'b']




#occ_lmo = ['F3_2s','F7_2s','F3_2pz','F7_2pz',
#'F3_2p2','F7_2p2','F3_2p1','F7_2p1']


occ_lmo = ['F3_2s','F7_2s','F3_2pz','F7_2pz']


lmo_vir = ["F3_2pz","F7_2pz","F3_3pz","F7_3pz","F3_3s","F7_3s","F3_3dz",
"F7_3dz","F3_3py","F7_3py","F3_3px","F7_3px","F3_3dxy","F7_3dxy","F3_3dx2-y2","F7_3dx2-y2"
,"F3_3dyz","F7_3dyz","F3_3dxz", "F7_3dxz", 'V_CC' ]

lmo_vir = ["F3_2pz","F7_2pz","F3_3pz","F7_3pz","F3_3s","F7_3s" ]


#df_F_C = data_J[(data_J.i.str.contains('F3_2s') == True) & (data_J.j.str.contains('F7_2s') == True) & 
#(data_J.a.str.contains("F3_2pz") == True) & (data_J.b.str.contains("F3_2pz") == True)].reset_index()
#df_F_C.fc_ab = 0
for i in occ_lmo:
    for j in occ_lmo:
        for a in lmo_vir:
            for b in lmo_vir: 
                df = data_J[(data_J.i.str.contains(i) == True) & (data_J.j.str.contains(j) == True)
                     & (data_J.a.str.contains(a) == True) & (data_J.b.str.contains(b) == True)]
                if df.fc_ab[abs(df.fc_ab) > 3].any():
                    #ang = df.ang
                    #df_F_C.fc_ab += df.reset_index().fc_ab
                    print(i,a,j,b)
                    #print(df_F_C.fc_ab)
#plt.figure(figsize=(10,8))
#plt.plot(df.ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
#plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')

#plt.legend()
#plt.ylabel('Hz')
#plt.xlabel('Ángulo diedro')
#plt.suptitle('FC contribution to ^3J(F-F)_{ia,jb}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_C2F2H4_{i}_{a}_{j}_{b}.png', dpi=200)

plt.show()                #


