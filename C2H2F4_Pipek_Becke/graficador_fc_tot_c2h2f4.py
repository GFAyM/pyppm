import matplotlib.pyplot as plt
import pandas as pd

text = 'pathways_fc_iajb_c2h2f4_all.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang','fc_ij', 'fc_iajb', 'a', 'b']

lmo_vir = ['C-H_1','C-H_2', 'H1_2pz','H2_2pz', 'H1_1s', 'H2_1s', 'H1_2s', 'H2_2s','H1_2px', 'H2_2px','H1_2py','H2_2py']
lmo_vir = ['C-H_1','C-H_2', 'H1_2pz','H2_2pz', 'H1_2s', 'H2_2s']


df_F_C = data_J[(data_J.a.str.contains('H1_2pz') == True) & (data_J.b.str.contains('H2_2pz') == True)].reset_index()
df_F_C.fc_iajb = 0

for orb1 in lmo_vir:
    for orb2 in lmo_vir:

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C.fc_iajb += df.reset_index().fc_iajb

        

text = 'pathways_fc_c2h2f4.txt'
data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang','fc_tot', 'fc', 'a', 'b']
df_F_C_occ = data_J[(data_J.a.str.contains('O-H1') == True) & (data_J.b.str.contains('O-H1') == True)].reset_index()
df_F_C_occ.fc = 0

occ_lmo = ['O-H1','O-H2']

for orb1 in occ_lmo:
    for orb2 in occ_lmo:
        
#        df_F_C = data_J[(data_J.a.str.contains('F3_2pz') == True) & (data_J.b.str.contains('F7_2pz') == True)]

        df = data_J[(data_J.a.str.contains(orb1) == True) & (data_J.b.str.contains(orb2) == True)]
        ang = df.ang
        df_F_C_occ.fc += df.reset_index().fc

plt.figure(figsize=(10,8))

plt.plot(df_F_C_occ.ang, df_F_C_occ.fc, 'b>-', label=r'J$_{ij}$')#f'a={orb1} b={orb2}')
plt.plot(df_F_C.ang, df_F_C.fc_iajb, 'g<-', label=r'$J_{ia,jb}$')
plt.plot(df_F_C.ang, df_F_C_occ.fc_tot, 'm<-', label='J')
plt.legend()
plt.title(r'Occupied and virtual LMOs contribution to $^{FC}J$(H-H) in C$_2$H$_2$F$_4$')
plt.savefig('FC_occ_vir_C2H2F4.png')
plt.show()    

