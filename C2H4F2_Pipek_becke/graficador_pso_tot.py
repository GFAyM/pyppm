import matplotlib.pyplot as plt
import pandas as pd


lmo_vir = ["F3_2pz","F7_2pz","F3_3pz","F7_3pz","F3_3s","F7_3s","F3_3py","F7_3py","F3_3px","F7_3px"]

text = 'cloppa_pso_ij_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'pso', 'a', 'b']

occ_lmo = ['F3_2pz','F7_2pz','F3_LPx','F7_LPx','F3_LPy','F7_LPy']#,'F3_2s', 'F7_2s']

df_F_C_occ = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_occ.pso = 0

for orb1 in occ_lmo:
    for orb2 in occ_lmo:
        
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.pso[abs(df.pso) > 0].any():
            ang = df.ang
            df_F_C_occ.pso += df.reset_index().pso


text = 'cloppa_pso_iajb_C2H4F2_2pxyz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso', 'a', 'b']

df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F3_2pz') == True)].reset_index()
df_F_C.pso = 0

for orb1 in lmo_vir:
    for orb2 in lmo_vir:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C.pso += df.reset_index().pso
        
            #df_F_C.pso += df.reset_index().pso


data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
data_J = pd.DataFrame(data_J)
PSO = data_J[3]

plt.figure(figsize=(10,8))

plt.plot(df_F_C_occ.ang, df_F_C_occ.pso, 'b>-', label=r'J$_{ij}$')#f'a={orb1} b={orb2}')
plt.plot(df_F_C.ang, df_F_C.pso, 'g<-', label=r'$J_{ia,jb}$')
plt.plot(df_F_C.ang, PSO, 'm<-', label='J')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('Occupied and virtual contribution to $^{PSO}J(F-F)$')
#plt.title('')# f'a={orb1}, b={orb2}')
plt.savefig(f'PSO_occ_vir.png', dpi=200)
plt.show()                #


#data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
#data_J = pd.DataFrame(data_J)


#ang = data_J[0]
#PSO = data_J[3]


