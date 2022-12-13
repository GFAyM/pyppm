import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'elementos_fcsd_c2h4f2_iajb.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang','p1','p2','m','i','a','j','b']




#occ_lmo = ['F3_2pz', 'F7_2pz', 'F3_2s', 'F7_2s']

occ_lmo = ['F3_2pz', 'F7_2pz','F3_2p2', 'F7_2p2']

occ_lmo1 = ['F3_2pz','F3_2p']
occ_lmo2 = ['F7_2pz','F7_2p']


lmo_vir = ["F3_2pz","F7_2pz", 'F3_3s', 'F7_3s']

lmo_vir1 = ["F3_2pz","F3_3s"]
lmo_vir2 = ["F7_2pz","F7_3s"]


df_F_C = data_J[(data_J.i.str.contains('F3_2pz') == True) & (data_J.j.str.contains('F7_2pz') == True) & 
(data_J.a.str.contains("F3_2pz") == True) & (data_J.b.str.contains("F3_2pz") == True)].reset_index()

for i in occ_lmo1:
    for j in occ_lmo2:
        for a in lmo_vir1:
            for b in lmo_vir2: 
                df = data_J[(data_J.i.str.contains(i) == True) & (data_J.j.str.contains(j) == True)
                     & (data_J.a.str.contains(a) == True) & (data_J.b.str.contains(b) == True)]
                #if df.fc_ab[abs(df.fc_ab) > 3].any():
                #ang = df.ang
                #df_F_C.p1 += df.reset_index().p1
                #df_F_C.p2 += df.reset_index().p2
                #df_F_C.m += df.reset_index().m                
#                    print(i,a,j,b)
                #print(df_F_C.fc_ab)
                #print(df)
                ii=df[((df.ang == 120))].index
                df = df.drop(ii)
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14,8))
                #plt.figure(figsize=(10,8))

                ax1.plot(df.ang, df.p1, 'b>-', label=f'i={i} a={a}') #f'a={orb1} b={orb2}')
                ax1.set_title(r'${b}^{FC}_{F_3,ia}$')
                ax1.legend()
                ax3.plot(df.ang, df.p2, 'b>-', label=f'i={j} a={b}' )#f'a={orb1} b={orb2}')
                ax3.set_title(r'${b}^{FC}_{F_7,jb}$')
                ax3.legend()
                ax2.plot(df.ang, df.m, 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
                ax2.set_title(r'$^3{P}_{ia,jb}$')
                #plt.ylabel('Hz')
                ax1.set_xlabel('Ángulo diedro')
                ax2.set_xlabel('Ángulo diedro')
                ax3.set_xlabel('Ángulo diedro')

                ax4.plot(df.ang, df.p1*df.p2*df.m, 'r>-')
                ax4.set_title(r'${b}^{FC}_{F_3,ia}$* $^3{P}_{ia,jb}$* $^3{P}_{ia,jb}$')
                ax4.set_xlabel('Angulo diedro')
                plt.suptitle(r'''Elements of Polarization Propagator $J^{FC}$ in C$_2$H$_4$F$_2$''')
        
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
                plt.savefig(f'FC_elements_C2F2H4_{i}_{a}_{j}_{b}.png', dpi=200)
                plt.show()   


