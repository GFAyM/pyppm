import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_vir_C2H6_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']






#vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s', 'H3_2px', 'H7_2px','H3_2py', 'H7_2py', 'H3_2pz', 'H7_2pz']#, 

#vir_lmo =['H3_1s', 'H7_1s','H3_2s', 'H7_2s', 'H3_2px', 'H7_2px',]#, ]#, ]#, 

vir_lmo1 = ['H3_1s', 'H3_2s', 'H3_2pz'] #, 'H3_2pz', 'H3_2s']
vir_lmo2 = ['H7_1s', 'H7_2s', 'H7_2pz'] #, 'H7_2pz', 'H7_2s']

df_F_C = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C.fc_ab = 0
for orb1 in vir_lmo1:
    for orb2 in vir_lmo2:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            #ang = df.ang
            df_F_C.fc_ab += df.reset_index().fc_ab
    #        fc_ab = fc_F_C + fc_ab_reit
            fc_ab = df.fc_ab
            #print(orb1,orb2)

#print(df_F_C)
plt.figure(figsize=(10,8))
plt.plot(df_F_C.ang, df_F_C.fc_ab, 'b^-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
#plt.plot(ang, DSO, 'm--', label='DSO')
plt.plot(df_F_C.ang, df_F_C.fc, 'go-', label='$^{FC}J_{ij}(H-H)$')
#plt.plot(ang, FCSD+FC+PSO, 'm--', label='Total')
#plt.plot(ang, FCSD, 'r+-', label='FC+SD')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('FC contribution to $^3J(H-H)_{ia,jb}$ en C$_2$H$_6$, cc-pVDZ')
plt.title('i=C-H$_1$, a=O-H$_1 (1s, 2s, 2p_z)$ j=C-H$_1$, a=O-H$_1 (1s, 2s, 2p_z)$')# f'a={orb1}, b={orb2}')
#plt.set_size_inches(6.5, 6.5)
plt.savefig(f'FC_vir_C2H6.png', dpi=200)
plt.show()               
