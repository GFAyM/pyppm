import matplotlib.pyplot as plt
import pandas as pd

text = 'cloppa_fc_iajb_C2H4F2.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'i', 'a', 'j', 'b']

lmo_occ = ['O-H1', 'O-H2']

lmo_vir = ["H1_1s","H2_1s","H1_2s","H2_2s","H1_2px","H2_2px","H1_2py","H2_2py","H1_2pz","H2_2pz"]


#df_F_C = data_J[(data_J.i.str.contains('O-H1') == True) & (data_J.j.str.contains('O-H2') == True) & 
#(data_J.a.str.contains("F3_2pz") == True) & (data_J.b.str.contains("F3_2pz") == True)].reset_index()

#df_F_C.fc_ab = 0

for i in lmo_occ:
    for j in lmo_occ:
        for a in lmo_vir:
            for b in lmo_vir: 
                df = data_J[(data_J.i.str.contains(i) == True) & (data_J.j.str.contains(j) == True)
                        & (data_J.a.str.contains(a) == True) & (data_J.b.str.contains(b) == True)].reset_index()


        #        if (df_p1.fc_ab + df_p2.fc_ab)[abs(df_p1.fc_ab + df_p2.fc_ab) > 0.5].any():
                if (df.fc_ab)[abs(df.fc_ab) > 0.3].any():


            #            print(df_F_C.fc_ab)
                    plt.figure(figsize=(10,8))
                    plt.plot(df.ang, df.fc_ab, 'b>-', label=r'$J_{ia,jb}$' )#f'a={orb1} b={orb2}')
                    #plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')

                    plt.legend()
                    plt.ylabel('Hz')
                    plt.xlabel('Ángulo diedro')
                    
                    plt.suptitle(r'$^{FC}J$(H-H) pathways in C$_2$H$_2$F$_4$ with cc-pVDZ base')
                    plt.title(f'i={i},a={a},j={j},b={b}')# f'a={orb1}, b={orb2}')
                    plt.savefig(f'FC_C2H2F4_{i}_{a}_{j}_{b}.png', dpi=200)
                    plt.close('all')
                    #plt.show()                #


