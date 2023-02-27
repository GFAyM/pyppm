import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fcsd_iajb_C2H4F2_2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fcsd', 'i', 'a', 'j', 'b']

occ_lmo1 = ['F3_2pz','F3_2s','F3_LPx','F3_LPy']
occ_lmo2 = ['F7_2pz','F7_2s','F7_LPx','F7_LPy']

lmo_vir1 = ["F3_2pz_","F3_3pz_","F3_3s_", "F3_3py_", "F3_3px_"]

lmo_vir2 = ["F7_2pz_","F7_3pz_","F7_3s_", "F7_3py_", "F7_3px_"]


df_F_C = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.j.str.contains('F3_2pz') == True) &
            (data_J.a.str.contains('F3_2pz_') == True) & (data_J.b.str.contains('F3_2pz_') == True)].reset_index()
df_F_C.fc_ab = 0

for orb_i in occ_lmo1:
    for orb_j in occ_lmo2:
        for orb_a in lmo_vir1:
            for orb_b in lmo_vir2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C.fc_ab += df.reset_index().fc_ab


fig, (ax1) = plt.subplots(1, 1, figsize=(10,10))
ax1.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
ax1.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{FC}J_{ij}(H-H)$')
ax1.set_ylabel('Hz')
ax1.set_xlabel('Dihedral angle')
ax1.set_title('i=j=[C-H1;C-H2], a=b=[O-H1*;O-H2*]')# f'a={orb1}, b={orb2}')

ax1.legend()



plt.suptitle('FC contribution to $^3J(H_1-H_2)_{ia,jb}$ in H$_2$C$_6$, using antiligants 1s y 2s.')


plt.savefig(f'FC_iajb_C2H6.png', dpi=200)


plt.show()               


