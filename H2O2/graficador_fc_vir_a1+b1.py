import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_vir_a1+b1_H2O2_631G.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']




#vir_lmo =['H3_1s', 'H4_1s', 'H3_2s', 'H4_2s', 'H3_2px', 'H4_2px','H3_2py', 'H4_2py', 'H3_2pz', 'H4_2pz']#, 
#vir_lmo =['H3_1s', 'H4_1s', 'H3_2s', 'H4_2s','H3_2px', 'H4_2px','H3_2py', 'H4_2py', 'H3_2pz', 'H4_2pz',
#'O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']#, 
vir_lmo = ['O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']

#'O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']

vir_lmo1 = ['H3_1s','H3_2s', 'H3_2px','H3_2pz', 'H3_2py']
vir_lmo2 = ['H4_1s','H4_2s', 'H4_2px','H4_2pz', 'H4_2py']


df_F_C = data_J[(data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C.fc_ab = 0
for orb1 in vir_lmo1:
    for orb2 in vir_lmo2:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.fc_ab[abs(df.fc_ab) > 0].any():
            ang = df.ang
            df_F_C.fc_ab += df.reset_index().fc_ab
    #        fc_ab = fc_F_C + fc_ab_reit
            a = df.a
            b = df.b
            #fc = df.fc
            fc_ab = df.fc_ab
            #print(orb1,orb2)

#print(df_F_C)
plt.figure(figsize=(10,8))
plt.plot(ang, df_F_C.fc_ab, 'b^-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
#plt.plot(ang, DSO, 'm--', label='DSO')
plt.plot(ang, df_F_C.fc, 'go-', label='$^{FC}J_{i,j}(H-H)$')
#plt.plot(ang, FCSD+FC+PSO, 'm--', label='Total')
#plt.plot(ang, FCSD, 'r+-', label='FC+SD')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('FC PZOA contribution to $^3J(H-H)_{ia,jb}$ en H$_2$O$_2$, 6-31G**')
plt.title('i=j=O-H. a=O-H_1* b=O-H_2*')# f'a={orb1}, b={orb2}')
#plt.set_size_inches(6.5, 6.5)
               
#plt.savefig(f'FC_a1+b1_vir_H2O2.png', dpi=200)
plt.show()

