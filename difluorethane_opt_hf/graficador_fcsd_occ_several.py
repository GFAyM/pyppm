import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fcsd_occ_C2F2H4_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']

#####PARA ANALIZAR
#data_J[(data_J.LMOS.str.contains('F3') == True) & (data_J.pp > 1)]
#data_J.LMOS +'*'+ data_J.LMOS
#orb = ['F3_2pz']
#for a in []

occ_lmo = ['H3_1s', 'H7_1s']

occ_lmo = ['F3_1s','F7_1s','F3_2s','F7_2s','F3_2pz','F7_2pz','F3_2p1','F7_2p1',
'F3_2p2','F7_2p2','C1_1s','C2_1s','C_C']

occ_lmo = ['F3_2s','F7_2s','F3_2pz','F7_2pz']

occ_lmo1 = ['F3_2s','F3_2pz']

occ_lmo2 = ['F7_2s','F7_2pz']

df_F_C = data_J[(data_J.a.str.contains('F3_1s') == True) & (data_J.b.str.contains('F7_1s') == True)].reset_index()
df_F_C.fc_ab = 0

for orb1 in occ_lmo1:
    for orb2 in occ_lmo2:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 5].any():
            ang = df.ang
            #df_F_C.fc_ab += df.reset_index().fc_ab
            
            plt.figure(figsize=(10,8))
            plt.plot(ang, df.fc_ab, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
            plt.plot(ang, df.fc, 'g<-', label='$^{FC}J(H-H)$')

            plt.legend()
            plt.ylabel('Hz')
            plt.xlabel('Ángulo diedro')
            plt.suptitle('FC+SD contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
            plt.title(f'i={orb1}, j={orb2}, a = b = all')# f'a={orb1}, b={orb2}')
            #plt.savefig(f'FC_occ_C2H6.png', dpi=200)
            plt.show()                #


