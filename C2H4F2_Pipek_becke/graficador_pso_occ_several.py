import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_pso_ij_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso', 'a', 'b']

occ_lmo = ['F3_2pz','F7_2pz','F3_LPx','F7_LPx','F3_LPy','F7_LPy']

occ_lmo_1 = ['F3_2pz','F3_LPx','F3_LPy']
occ_lmo_2 = ['F7_2pz','F7_LPx','F7_LPy']


df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F3_2pz') == True)].reset_index()
df_F_C.pso = 0

for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_1:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        if df.pso[abs(df.pso) > 1].any():
            ang = df.ang
            df_F_C.pso += df.reset_index().pso

data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
data_J = pd.DataFrame(data_J)


ang = data_J[0]
PSO = data_J[3]

plt.figure(figsize=(10,8))
plt.plot(ang, df_F_C.pso, 'b>-', label='$^{FC}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
plt.plot(ang, PSO, 'r>-', label='$^{FC}J(F-F)$' )#f'a={orb1} b={orb2}')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('PSO contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
plt.title(f'i={orb1}, j={orb2}, a = b = all')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_{orb1}_{orb2}.png', dpi=200)
plt.show()                #


