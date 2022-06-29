import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_vir_C2H6_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']


occ_lmo = ['H3_1s', 'H7_1s']




vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s', 'H3_2px', 'H7_2px','H3_2py', 'H7_2py', 'H3_2pz', 'H7_2pz']#, 

#vir_lmo =['H3_1s', 'H7_1s',  'H3_2px', 'H7_2px','H3_2pz', 'H7_2pz','H3_2py', 'H7_2py']#, ]#, 



for orb1 in vir_lmo:
    for orb2 in vir_lmo:
        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
                
        if df.fc_ab[abs(df.fc_ab) > 0.5].any():
        #        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

                ang = df.ang
                #df_F_C.fc_ab += df.reset_index().fc_ab
        #        fc_ab = fc_F_C + fc_ab_reit
                a = df.a
                b = df.b
                fc = df.fc
                fc_ab = df.fc_ab
                print(orb1,orb2)
                #plt.plot(ang, DSO, 'ro', label='DSO')
                plt.figure(figsize=(10,8))
                plt.plot(ang, df.fc_ab, 'b^-', label='$^{FC}J_{ia,jb}$ ' )#f'a={orb1} b={orb2}')
                #plt.plot(ang, fc, 'go-', label='$^{FC}J_{ij}$')
                #plt.plot(ang, FCSD+FC+PSO, 'm--', label='Total')
                #plt.plot(ang, FCSD, 'r+-', label='FC+SD')

                plt.legend()
                plt.ylabel('Hz')
                plt.xlabel('Ángulo diedro')
                plt.suptitle('FC contribution to $^3J(H-H)_{ia,jb}$ en C$_2$H$_6$, cc-pVDZ')
                plt.title(f'i={orb1}  j={orb2}')# f'a={orb1}, b={orb2}')
                #plt.set_size_inches(6.5, 6.5)
                plt.show()
                #plt.savefig(f'FC_sum_C2H4F2_ccpvdz.png', dpi=200)


        #fig = px.line(df, x="ang", y="pp")#, animation_frame='LMOS')
        #fig.update_layout(    yaxis_title=r'SSC Coupling [Hz]' )

        #fig.write_html("fc_coupling_pathways_C2H4F2.html", include_mathjax='cdn')
