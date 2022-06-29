import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pandas as pd

text = 'cloppa_fc_C2H4F2_ccpvdz.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'a', 'b', 'fc']

#####PARA ANALIZAR
#data_J[(data_J.LMOS.str.contains('F3') == True) & (data_J.pp > 1)]
#data_J.LMOS +'*'+ data_J.LMOS
#orb = ['F3_2pz']
#for a in []

orb = ['F3_2pz', 'F7_2pz', 'F3_3pz', 'F7_3pz', 'F3_3s', 'F7_3s', 'F3_3dz', 'F7_3dz', 
'F3_3py', 'F7_3py', 'F3_3px', 'F7_3px', 'F3_3dxy', 'F7_3dxy', 'F3_3dx2-y2', 
'F7_3dx2-y2', 'F3_3dyz', 'F7_3dyz', 'F3_3dxz', 'F7_3dxz', 'V_CC']



#orb  = ['F3_2pz', 'F7_2pz',  'F3_3pz', 'F7_3pz','F3_3py', 'F7_3py', 'F3_3px', 'F7_3px']


for orb1 in orb:
    for orb2 in orb:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
#        fc_F_C = df_F_C.reset_index().fc_ab
        if df.fc_ab[abs(df.fc_ab) > 3].any():
            ang = df.ang
    #        fc_ab_reit = df.reset_index().fc_ab
    #        fc_ab = fc_F_C + fc_ab_reit
            a = df.a
            b = df.b
            fc = df.fc
            fc_ab = df.fc_ab
            FC = 'FC'
            ia=str('ia')
            jb=str('jb')
            #plt.plot(ang, DSO, 'ro', label='DSO')
            plt.figure(figsize=(10,8))
            plt.plot(ang, fc_ab, 'b^-', label='$^{FC}J_{ia,jb}$ ' f'a={orb1} b={orb2}')
            #plt.plot(ang, DSO, 'm--', label='DSO')
            #plt.plot(ang, fc, 'go-', label='$^{FC}J_{ij}$')
            #plt.plot(ang, FCSD+FC+PSO, 'm--', label='Total')
            #plt.plot(ang, FCSD, 'r+-', label='FC+SD')

            plt.legend()
            plt.ylabel('Hz')
            plt.xlabel('Ángulo diedro')
            plt.suptitle('FC contribution to $^3J(F-F)_{ia,jb}$ en C$_2$H$_4$F$_2$, cc-pVDZ')
            plt.title('i=F3_2pz  j=F7_2pz' f'a={orb1}, b={orb2}')
            #plt.set_size_inches(6.5, 6.5)
            #plt.show()
            plt.savefig(f'FC_{orb1}_{orb2}_C2H4F2_ccpvdz.png', dpi=200)
        else:
            pass

        #fig = px.line(df, x="ang", y="pp")#, animation_frame='LMOS')
        #fig.update_layout(    yaxis_title=r'SSC Coupling [Hz]' )

        #fig.write_html("fc_coupling_pathways_C2H4F2.html", include_mathjax='cdn')
