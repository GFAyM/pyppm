import matplotlib.pyplot as plt
import pandas as pd

text = 'pathways_fc_iajb_c2h2f4.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fcsd', 'i', 'a', 'j', 'b']

#data_J_tot = pd.read_csv('mechanism_C2H2F4_ccpvdz.txt', sep='\s+', header=None)
#data_J_tot.columns = ['ang', 'fcsd', 'fc', 'pso']


lmo_vir1 = ["H3_2pz","H3_1s","H3_2s","H3_2px","H3_2py","C1_2pz"]
lmo_vir2 = ["H7_2pz","H7_1s","H7_2s","H7_2px","H7_2py","C2_2pz"]



lmo_vir = ["H3_2pz","H3_1s","H3_2s","H3_2px","H3_2py","C1_2pz","H7_2pz","H7_1s","H7_2s","H7_2px","H7_2py","C2_2pz" ]

lmo_occ = ['CH1', 'CH2']
lmo_occ1 = ['CH1']
lmo_occ2 = ['CH2']


df_F_C = data_J[(data_J.i.str.contains('CH1') == True) & (data_J.a.str.contains('H3_2pz') == True)
                & (data_J.j.str.contains('CH2') == True) & (data_J.b.str.contains('H7_2pz') == True)].reset_index()
#print(df_F_C)
df_F_C.fcsd = 0
#print(df_F_C)
for i in lmo_occ:
    for j in lmo_occ:
        for a in lmo_vir:
            for b in lmo_vir:
        
                df = data_J[(data_J.i.str.contains(i) == True) 
                          & (data_J.a.str.contains(a) == True) 
                          & (data_J.j.str.contains(j) == True)
                          & (data_J.b.str.contains(b) == True)]
#                print(df)
                if df.fcsd[abs(df.fcsd) > 1].any():
                    ang = df.ang
                    df_F_C.fcsd += df.reset_index().fcsd
                    #print(df_F_C)

plt.figure(figsize=(10,8))
plt.plot(ang, df_F_C.fcsd, 'b>-', label=f'{i}_{a}_{j}_{b}' )
#plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')
plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('FC+SD contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
#plt.title('i=C-F$_1$(2s,2p$_z$, 2p$_x$), j=C-F$_2$(2s,2p$_z$,2p$_x$), a = b = all')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)
plt.show()                #


