import matplotlib.pyplot as plt
import pandas as pd

text = 'cloppa_pso_iajb_C2H4F2_all.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso', 'a', 'b']

lmo_vir = ["F3_2pz","F7_2pz","F3_3pz","F7_3pz","F3_3s","F7_3s","F3_3py","F7_3py","F3_3px","F7_3px"]




df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F3_2pz') == True)].reset_index()
df_F_C.pso = 0

for orb1 in lmo_vir:
    for orb2 in lmo_vir:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C.pso += df.reset_index().pso
        
            #df_F_C.pso += df.reset_index().pso

plt.figure(figsize=(10,8))

plt.plot(df_F_C.ang, df_F_C.pso, 'b>-', label=f'a={orb1} b={orb2}' )#f'a={orb1} b={orb2}')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('PSO contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
#plt.title('')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_{orb1}_{orb2}.png', dpi=200)
plt.show()                #


#data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
#data_J = pd.DataFrame(data_J)


#ang = data_J[0]
#PSO = data_J[3]


