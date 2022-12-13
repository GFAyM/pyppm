import matplotlib.pyplot as plt
import pandas as pd

text = 'cloppa_fcsd_iajb_C2H4F2.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fcsd', 'i', 'a', 'j', 'b']

data_J_tot = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
data_J_tot.columns = ['ang', 'fcsd', 'fc', 'pso']



occ_lmo1 = ['F3_2pz','F3_2s']
occ_lmo2 = ['F7_2pz','F7_2s']


lmo_vir1 = ["F3_2pz_","F3_3pz_","F3_3s_", "F3_3py_", "F3_3px_"]

lmo_vir2 = ["F7_2pz_","F7_3pz_","F7_3s_", "F7_3py_", "F7_3px_"]

df_F_C = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.a.str.contains('F3_2pz_') == True)
                & (data_J.j.str.contains('F7_2pz') == True) & (data_J.b.str.contains('F7_2pz_') == True)].reset_index()
#print(df_F_C)
df_F_C.fcsd = 0
#print(df_F_C)
for i in occ_lmo1:
    for j in occ_lmo2:
        for a in lmo_vir1:
            for b in lmo_vir2:
        
                df = data_J[(data_J.i.str.contains(i) == True) 
                          & (data_J.a.str.contains(a) == True) 
                          & (data_J.j.str.contains(j) == True)
                          & (data_J.b.str.contains(b) == True)]

                if df.fcsd[abs(df.fcsd) > 3].any():
                    ang = df.ang
                    
                    #print(df.fcsd)
                    #print(df.reset_index().fcsd)
                    #df_F_C.fcsd += df.reset_index().fcsd
                    #b = df.b
                    #fc = df.fc
                    #fc_ab = df.fc_ab
                    plt.figure(figsize=(10,8))
                    plt.plot(df_F_C.ang, df.fcsd, 'b>-', label=f'{i}_{a}_{j}_{b}' )#f'a={orb1} b={orb2}')
                    #plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')
                    plt.legend()
                    plt.ylabel('Hz')
                    plt.xlabel('Ángulo diedro')
                    plt.suptitle('FC+SD contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
                    #plt.title('i=C-F$_1$(2s,2p$_z$, 2p$_x$), j=C-F$_2$(2s,2p$_z$,2p$_x$), a = b = all')# f'a={orb1}, b={orb2}')
                    #plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)
                    plt.show()                #


