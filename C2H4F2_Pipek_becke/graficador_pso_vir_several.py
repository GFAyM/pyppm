import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_pso_iaia_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso', 'a', 'b']

occ_lmo = ['F3_2pz','F7_2pz','F3_LPx','F7_LPx','F3_LPy','F7_LPy']

occ_lmo_1 = ['F3_2pz','F3_LPx','F3_LPy']
occ_lmo_2 = ['F7_2pz','F7_LPx','F7_LPy']

lmo_vir1 = ["F3_2pz_","F3_3pz_","F3_3s_","F3_3py_","F3_3px_"]
lmo_vir2 = ["F7_2pz_","F7_3pz_","F7_3s_","F7_3py_","F7_3px_"]

lmo_vir = ["F3_2pz","F3_3pz","F3_3s","F3_3py","F3_3px","F7_2pz","F7_3pz","F7_3s","F7_3py","F7_3px"]


df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C.pso = 0

for orb1 in lmo_vir1:
    for orb2 in lmo_vir1:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.pso[abs(df.pso) > 2].any():
            
            #df_F_C.pso += df.reset_index().pso
            plt.figure(figsize=(10,8))

            plt.plot(df_F_C.ang, df.pso, 'b>-', label=f'a={orb1} b={orb2}' )#f'a={orb1} b={orb2}')

            plt.legend()
            plt.ylabel('Hz')
            plt.xlabel('Ángulo diedro')
            plt.suptitle('PSO contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
            plt.title(f'i={orb1}, j={orb2}, a = b = all')# f'a={orb1}, b={orb2}')
            #plt.savefig(f'FC_occ_C2F2H4_{orb1}_{orb2}.png', dpi=200)
            plt.show()                #


#data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
#data_J = pd.DataFrame(data_J)


#ang = data_J[0]
#PSO = data_J[3]


