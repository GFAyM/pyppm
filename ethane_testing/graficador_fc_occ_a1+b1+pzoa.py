import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_a1+b1_occ_C2H6_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']

text_pzoa = 'cloppa_fc_pzoa_occ_C2H6_ccpvdz.txt'
data_J_pzoa = pd.read_csv(text, sep='\s+', header=None)

data_J_pzoa.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']

#####PARA ANALIZAR
#data_J[(data_J.LMOS.str.contains('F3') == True) & (data_J.pp > 1)]
#data_J.LMOS +'*'+ data_J.LMOS
#orb = ['F3_2pz']
#for a in []

occ_lmo = ['H3_1s', 'H7_1s']

occ_lmo1 = ['H7_1s']
occ_lmo2 = ['H3_1s']

df_F_C = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H7_1s') == True)].reset_index()
df_F_C.fc_ab = 0



for orb1 in occ_lmo:
    for orb2 in occ_lmo:
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
#        if df.fc_ab[abs(df.fc_ab) > 0].any():
        df_pzoa = data_J_pzoa[(data_J_pzoa.a.str.contains(orb1) == True) & (data_J_pzoa.b.str.contains(orb2) == True)]
        
        ang = df.ang
        df_F_C.fc_ab += df.reset_index().fc_ab
        df_F_C.fc_ab += df_pzoa.reset_index().fc_ab
        
        df_F_C.fc += df.reset_index().fc
        df_F_C.fc += df_pzoa.reset_index().fc
        

        #fc_ab = df.fc_ab

plt.figure(figsize=(10,8))
plt.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
plt.plot(ang, df_F_C.fc, 'g<-', label='$^{FC}J(H-H)$')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('FC contribution to $^3J(H-H)_{i,j}$(A(1)+B(1) aproach) in C$_2$O$_2$, cc-pVDZ')
plt.title('i=C-H$_1$, j=C-H$_2$, a = b = all')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2H6.png', dpi=200)
plt.show()                #


