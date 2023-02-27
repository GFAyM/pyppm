import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_pso_iajb_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso', 'a', 'b']


lmo_vir1 = ["F3_2pz_","F3_3pz_","F3_3s_","F3_3py_","F3_3px_"]

lmo_vir2 = ["F7_2pz_","F7_3pz_","F7_3s_","F7_3py_","F7_3px_"]





df_F_C = data_J[(data_J.a.str.contains('F3_2pz_') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
df_F_C.pso = 0

for orb1 in lmo_vir1:
    for orb2 in lmo_vir2:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C.pso += df.reset_index().pso
        

text = 'cloppa_pso_iaia_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso', 'a', 'b']

df_F_C_1 = data_J[(data_J.a.str.contains('F3_2pz_') == True) & (data_J.b.str.contains('F3_2pz_') == True)].reset_index()
df_F_C_1.pso = 0

for orb1 in lmo_vir1:
    for orb2 in lmo_vir1:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C_1.pso += df.reset_index().pso


text = 'cloppa_pso_ij_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'pso', 'a', 'b']
occ_lmo_1 = ['F3_2pz','F3_LPx','F3_LPy']#,'F3_2s']

occ_lmo_2 = ['F7_2pz','F7_LPx','F7_LPy']#, 'F7_2s']

df_F_C_2 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_2.pso = 0

for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_1:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C_2.pso += df.reset_index().pso

df_F_C_3 = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)].reset_index()
df_F_C_3.pso = 0
for orb1 in occ_lmo_1:
    for orb2 in occ_lmo_2:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C_3.pso += df.reset_index().pso



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
ax1.plot(df_F_C.ang, df_F_C.pso, 'b>-', label=r'$^{PSO}J_{iajb}$' )#f'a={orb1} b={orb2}')
ax1.plot(df_F_C_3.ang, df_F_C_3.pso, 'g>-', label=r'$^{PSO}J_{ij}$' )#f'a={orb1} b={orb2}')

ax1.legend()
ax1.set_ylabel('Hz')
ax1.set_xlabel('Ángulo diedro')
ax1.set_title(f'iajb')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_{orb1}_{orb2}.png', dpi=200)

ax2.plot(df_F_C_1.ang, df_F_C_1.pso, 'b>-', label=r'$^{PSO}J_{iaia}$' )#f'a={orb1} b={orb2}')
ax2.plot(df_F_C_2.ang, df_F_C_2.pso, 'g>-', label=r'$^{PSO}J_{ii}$' )#f'a={orb1} b={orb2}')

ax2.legend()
ax2.set_ylabel('Hz')
ax2.set_xlabel('Ángulo diedro')
#ax2.suptitle('PSO contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
ax2.set_title(f'iaia')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_{orb1}_{orb2}.png', dpi=200)
plt.suptitle(r'Virtuals LMOs contribution to $^{PSO}J(F-F)_{ij}$')

plt.savefig(f'PSO_occ_C2F2H4_iajb_2col.png', dpi=200)
plt.show()                #


#data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
#data_J = pd.DataFrame(data_J)


#ang = data_J[0]
#PSO = data_J[3]


