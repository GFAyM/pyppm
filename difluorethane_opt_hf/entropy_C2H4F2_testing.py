import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.ppe import inverse_principal_propagator
from src.ppe import M_matrix
from src.cloppa import Cloppa

from src.help_functions import extra_functions
import plotly.express as px
import pandas as pd
import numpy as np
from pyscf import ao2mo

F3_2s = [11, 10, 10, 11, 10, 10, 11, 10, 11, 10, 11, 10, 11, 10, 10, 11, 10, 11, 11]
F7_2s = [10, 11, 11, 10, 11, 11, 10, 11, 10, 11, 10, 11, 10, 11, 11, 10, 11, 10, 10]


F3_2pz = [8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
F7_2pz = [9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]


occ_lmo = [(F3_2pz,'F3_2pz'), (F7_2pz,'F7_2pz'), (F3_2s,'F3_2p2'), (F7_2s,'F7_2p2')]

occ_lmo = [(F3_2pz,'F3_2pz'), (F7_2pz,'F7_2pz')]


v1_1= [73, 74, 74, 73, 73, 73, 73, 73, 74, 74, 73, 73, 73, 73, 73, 73, 73, 74, 74]
       #0   1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18 
v1_2= [74, 56, 73, 74, 75, 61, 75, 75, 73, 73, 74, 75, 75, 75, 74, 74, 75, 73, 73]

v2_1= [21, 21, 21, 20, 21, 21, 21, 21, 21, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21]
v2_2= [22, 22, 22, 21, 25, 23, 22, 22, 22, 21, 22, 22, 23, 23, 25, 22, 22, 22, 22]


v3_1= [41, 40, 39, 40, 40, 40, 39, 40, 40, 41, 40, 40, 39, 40, 40, 40, 40, 40, 40]
v3_2= [40, 41, 40, 41, 44, 45, 43, 41, 39, 40, 41, 41, 43, 41, 41, 41, 41, 41, 41]


v4_1= [64, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64]
       #0   1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18
v4_2= [63, 64, 64, 65, 62, 69, 69, 69, 69, 69, 65, 69, 69, 62, 67, 65, 64, 64, 63]

v6_1= [50, 50, 51, 51, 50, 51, 51, 50, 49, 50, 48, 49, 49, 51, 50, 50, 50, 49, 50]
v6_2= [49, 48, 48, 49, 48, 50, 50, 49, 50, 49, 51, 50, 51, 49, 51, 51, 49, 50, 49]


#falta agregar los orbitales 3dy. 
v5_1= [43, 45, 45, 43, 43, 46, 45, 45, 45, 45, 46, 45, 45, 45, 44, 44, 44, 45, 45]
       #0   1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18 
v5_2= [45, 51, 50, 50, 41, 41, 47, 48, 51, 46, 45, 48, 47, 46, 46, 49, 48, 51, 44]

#además, agregar los 3dx y 3dy

v7_1= [68, 67, 67, 67, 68, 66, 67, 68, 67, 68, 68, 68, 67, 68, 66, 67, 68, 67, 68]
v7_2= [67, 69, 69, 69, 65, 68, 68, 67, 66, 65, 66, 67, 68, 66, 69, 69, 70, 69, 67]

v8_1= [69, 71, 71, 71, 70, 71, 71, 71, 71, 70, 70, 71, 71, 70, 71, 71, 71, 71, 70]
v8_2= [70, 70, 70, 70, 66, 67, 65, 64, 64, 64, 64, 64, 65, 65, 70, 70, 69, 70, 69]

v9_1= [62, 62, 62, 62, 61, 62, 62, 62, 62, 62, 61, 62, 62, 61, 62, 62, 62, 62, 61]
v9_2= [61, 68, 65, 64, 64, 64, 64, 65, 68, 61, 62, 65, 64, 64, 64, 64, 65, 68, 62]

v10_1= [66, 65, 66, 66, 67, 65, 66, 66, 65, 66, 67, 66, 66, 67, 65, 66, 66, 65, 66]
v10_2= [65, 66, 68, 68, 69, 70, 70, 70, 70, 67, 69, 70, 70, 69, 68, 68, 67, 66, 65]


lmo_vir = [(v1_1,"F3_2pz"),(v1_2,"F7_2pz"),(v2_1,"F3_3pz"),(v2_2,"F7_3pz"), (v3_1,"F3_3s"),(v3_2,"F7_3s")]

lmo_vir_1 = [(v1_1,"F3_2pz"),(v2_1,"F3_3pz"), (v3_1,"F3_3s")]
lmo_vir_2 = [(v1_2,"F7_2pz"),(v2_2,"F7_3pz"), (v3_2,"F7_3s")]


data = []
for ang in range(0,1,1): 
    mol, mo_coeff, mo_occ = extra_functions(molden_file=f"difluorethane_cc-pvdz_{ang*10}_Cholesky_PM.molden").extraer_coeff
    inv_prop = M_matrix(mol=mol, mo_coeff=mo_coeff, 
                occ = [ F3_2pz[ang], F3_2s[ang]],
                vir = [v1_1[ang], v2_1[ang]])
    m = inv_prop.m_iajb
    #inv_prop_old = inverse_principal_propagator(mol=mol, mo_coeff=mo_coeff, o1 = [F3_2pz[ang]], o2 = [F7_2pz[ang]],
    #                v1 = [v1_1[ang], v2_1[ang]], v2 = [v1_2[ang], v2_2[ang]])
    #eig = inv_prop.entropy
    #print(eig)
    #print(inv_prop_old.m_iaia)
    cloppa_obj = Cloppa(
		mo_coeff_loc=mo_coeff, mol_loc=mol, #vir=viridx, occ=occidx,
		mo_occ_loc=mo_occ)
    m = cloppa_obj.M(energy_m=False)
    m = m.reshape((cloppa_obj.nocc,cloppa_obj.nvir,cloppa_obj.nocc,cloppa_obj.nvir))
    nocc = cloppa_obj.nocc
    #o1 = mo_coeff[:,[F3_2pz[ang]]]
    #v1 = mo_coeff[:,[v1_1[ang]]]
    #v2 = mo_coeff[:,[v2_1[ang]]]
    #int = -ao2mo.general(mol, [v2, v2, o1, o1], compact=False)
    #int -= ao2mo.general(mol, [o1, v2, o1, v2], compact=False)
    #print(int)
    print(m[F3_2pz[ang],[v1_1[ang]-nocc],F3_2pz[ang],[v2_1[ang]-nocc]])


#df = pd.DataFrame(data, columns=['angulo', 'Entanglement', 'Virtuals',  'Mutual Information'])
#fig = px.line(df, x="angulo", y="Entanglement", animation_frame='Virtuals', color='Mutual Information',
#       title="Entropy iajb in the Difluorethane (opt-HF) using combinations of 2 virtuals as antibondings, and base cc-pvdz",
#      )
#fig.update_layout(    yaxis_title=r'Entrelazamiento' )
#fig.show()
#fig.write_html("m_iajb_difluorethane_2exc.html", include_mathjax='cdn')


