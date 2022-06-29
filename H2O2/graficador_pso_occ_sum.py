import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_occ_H2O2_631G.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']



occ_lmo = ['O1_1s', 'O2_1s',  'O1_2p1','O2_2p1','O1_2p2', 'O2_2p2','H4_1s','H3_1s','C_C']

occ_lmo = ['H4_1s','H3_1s', 'C_C']


df_F_C = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H4_1s') == True)].reset_index()
df_F_C.fc_ab = 0
for orb1 in occ_lmo:
    for orb2 in occ_lmo:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C.fc_ab += df.reset_index().fc_ab
    #        fc_ab = fc_F_C + fc_ab_reit
            a = df.a
            b = df.b
            fc = df.fc
            fc_ab = df.fc_ab
            print(orb1,orb2)


plt.figure(figsize=(10,8))
plt.plot(ang, df_F_C.fc_ab, 'b^-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
#plt.plot(ang, DSO, 'm--', label='DSO')
plt.plot(ang, fc, 'go-', label='$^{FC}J(H-H)$')
#plt.plot(ang, FCSD+FC+PSO, 'm--', label='Total')
#plt.plot(ang, FCSD, 'r+-', label='FC+SD')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('FC contribution to $^3J(H-H)_{ia,jb}$ en H$_2$O$_2$, 6-31G**')
#plt.title('i=F3_2pz  j=F7_2pz')# f'a={orb1}, b={orb2}')
#plt.set_size_inches(6.5, 6.5)
plt.show()                #plt.savefig(f'FC_sum_C2H4F2_ccpvdz.png', dpi=200)


