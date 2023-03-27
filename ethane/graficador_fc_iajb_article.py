import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

###this graph will going to show all the contribution of the occupied and virtual LMOs to the FC J-coupling 
# between H nuclei in ethane

text = 'cloppa_fc_iajb_C2H6_ccpvdz.txt'

data_J = pd.read_csv(text, sep='\s+', header=None)

data_J.columns = ['ang', 'fc_ab', 'i', 'a', 'j', 'b', 'fc_ij']

occ_lmo = ['lig_1','lig_2']
occ_lmo1 = ['lig_1']
occ_lmo2 = ['lig_2']

vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s', 'H3_2px', 'H7_2px','H3_2py', 'H7_2py', 'H3_2pz', 'H7_2pz']
#vir_lmo =['H3_1s', 'H7_1s', 'H3_2s', 'H7_2s', 'H3_2pz', 'H7_2pz']


vir_lmo1 =['H3_1s','H3_2s','H3_2px','H3_2py','H3_2pz']
vir_lmo1 =['H3_1s','H3_2s']

vir_lmo2 =['H7_1s','H7_2s','H7_2px','H7_2py','H7_2pz']
vir_lmo2 =['H7_1s','H7_2s']

text = 'ssc_mechanism_C2H6_ccpvdz.txt'
data_tot_J = pd.read_csv(text, sep='\s+', header=None)
data_tot_J.columns = ['ang', 'fcsd', 'fc', 'pso']


df_F_C = data_J[(data_J.i.str.contains('lig_1') == True) & (data_J.j.str.contains('lig_1') == True) &
            (data_J.a.str.contains('H3_1s') == True) & (data_J.b.str.contains('H3_1s') == True)].reset_index()
df_F_C.fc_ab = 0

for orb_i in occ_lmo1:
    for orb_j in occ_lmo2:
        for orb_a in vir_lmo1:
            for orb_b in vir_lmo2:
                df = data_J[(data_J.i.str.contains(orb_i) == True) & (data_J.j.str.contains(orb_j) == True) &
                    (data_J.a.str.contains(orb_a) == True) & (data_J.b.str.contains(orb_b) == True)]
                ang = df.ang
                df_F_C.fc_ab += df.reset_index().fc_ab




plt.figure(figsize=(10,8))
plt.plot(ang, df_F_C.fc_ab, 'b>-', label='$^{FC}J_{ia,jb}(H-H)$' )#f'a={orb1} b={orb2}')
plt.plot(ang, df_F_C.fc_ij, 'g<-', label='$^{FC}J(H-H)$')
plt.ylabel('Hz')
plt.xlabel('Dihedral angle')
plt.title('J(H-H) using principal coupling pathways')

plt.legend()



#plt.suptitle('FC contribution to $^3J(H_1-H_2)_{ia,jb}$ in H$_2$C$_6$, using antiligants 1s y 2s.')


plt.savefig(f'FC_iajb_C2H6_paper.png', dpi=200)


plt.show()               


