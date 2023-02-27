import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_ab_H2O2.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ij', 'fc_ab', 'a', 'b']




vir_lmo =['H1_1s', 'H2_1s', 'H1_2s', 'H2_2s', 'H1_2px', 'H2_2px','H1_2py', 'H2_2py', 'H1_2pz', 'H2_2pz']#, 
vir_lmo =['H1_1s', 'H2_1s', 'H1_2s', 'H2_2s', 'H1_2px', 'H2_2px']

# 'O1_3dz', 'O2_3dz', , 'O1_3py', 'O2_3py', ,'O1_3ddxz','O2_3ddxz']#, 
# 'O1_3dx', 'O2_3dx','O1_3dy', 'O2_3dy','O1_3s', 'O2_3s'
#vir_lmo = ['H3_2s', 'H4_2s', 'H3_2px', 'H4_2px','H3_2py', 'H4_2py', 'H3_2pz', 'H4_2pz','O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']

#'O1_3dz', 'O2_3dz', 'O1_3s', 'O2_3s', 'O1_3dx', 'O2_3dx', 'O1_3py', 'O2_3py', 'O1_3dy', 'O2_3dy','O1_3ddxz','O2_3ddxz']

vir_lmo1 = ['H1_1s','H1_2s', 'H1_2px','H1_2pz']#,'H1_2py','H1_2pz']
vir_lmo2 = ['H2_1s','H2_2s', 'H2_2px']#,'H2_2py','H2_2pz']

#vir_lmo1 = ['H3_1s','H3_2px','H3_2py','H3_2pz']
#vir_lmo2 = ['H4_1s','H4_2px','H4_2py','H4_2pz']


df_F_C = data_J[(data_J.a.str.contains('H1_1s') == True) & (data_J.b.str.contains('H1_1s') == True)].reset_index()
df_F_C.fc_ab = 0
for orb1 in vir_lmo1:
    for orb2 in vir_lmo1:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        #if df.fc_ab[abs(df.fc_ab) > 0].any():
        ang = df.ang
        df_F_C.fc_ab += df.reset_index().fc_ab
#        fc_ab = fc_F_C + fc_ab_reit
        #fc = df.fc
        #fc_ab = df.fc_ab
        #print(orb1,orb2)
print(df_F_C)

#print(df_F_C)
plt.figure(figsize=(10,8))
plt.plot(ang, df_F_C.fc_ab, 'b^-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
#plt.plot(ang, DSO, 'm--', label='DSO')
plt.plot(ang, df_F_C.fc_ij, 'go-', label='$^{FC}J_{ij}(H-H)$')
#plt.plot(ang, FCSD+FC+PSO, 'm--', label='Total')
#plt.plot(ang, FCSD, 'r+-', label='FC+SD')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('FC contribution to $^3J(H-H)_{ia,jb}$ en H$_2$O$_2$, 6-31G**')
plt.title('i=j=O-H. a=O-H_1* b=O-H_2*')# f'a={orb1}, b={orb2}')
#plt.set_size_inches(6.5, 6.5)
#plt.savefig(f'FC_vir_H2O2_2.png', dpi=200)

plt.show()               


