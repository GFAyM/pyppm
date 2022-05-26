import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.ppe import inverse_principal_propagator 

from src.help_functions import extra_functions
import itertools
import plotly.express as px
import pandas as pd
import numpy as np


o1= [8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
       #0   1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18 

o2= [9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]


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


V = [(v2_1,v2_2,"F3pz"),(v3_1,v3_2,"F3s"),(v4_1,v4_2,"F3dz"), (v5_1,v5_2,"F3py"),(v6_1,v6_2,"F3px"),
       (v7_1,v7_2,"F3dxy"), (v8_1,v8_2,"F3dx2-y2"), (v9_1,v9_2,"F3dyz"), (v10_1,v10_2, "F3dxz") ]

#(v1_1,v1_2,"Csp3-F2pz"),

data = []
for ang in range(0,19,1):
    for i in itertools.combinations(V,1):
       #print(v1_1[ang], i[0][1][ang])
       mol, mo_coeff, mo_occ = extra_functions(molden_file=f"difluorethane_cc-pvdz_{ang*10}_Cholesky_PM.molden").extraer_coeff
       inv_prop = inverse_principal_propagator(mol=mol, mo_coeff=mo_coeff, o1=[o1[ang]], o2=[o2[ang]], 
                                   v1=[v1_1[ang], i[0][0][ang]],
                                   v2=[v1_2[ang], i[0][1][ang]])
       mutual_information = -inv_prop.m_iajb
       mutual_information = np.diag(mutual_information).sum()
       data.append([ang*10, mutual_information, f'Csp3-F2pz_{i[0][2]}', "Entropía"])

df = pd.DataFrame(data, columns=['angulo', 'Entanglement', 'Virtuals',  'Mutual Information'])
fig = px.line(df, x="angulo", y="Entanglement", animation_frame='Virtuals', color='Mutual Information',
       title="Entropy iajb in the Difluorethane (opt-HF) using combinations of 2 virtuals as antibondings, and base cc-pvdz",
      )
fig.update_layout(    yaxis_title=r'Entrelazamiento' )
fig.show()
fig.write_html("m_iajb_triplet_difluorethane_2exc_2.html", include_mathjax='cdn')


data = []
for ang in range(0,19,1):
    for i in itertools.combinations(V,1):
        mol, mo_coeff, mo_occ = extra_functions(molden_file=f"difluorethane_cc-pvdz_{ang*10}_Cholesky_PM.molden").extraer_coeff
        inv_prop = inverse_principal_propagator(mol=mol, mo_coeff=mo_coeff, spin_dependence="singlet",
                                                 o1=[o1[ang]], o2=[o2[ang]], 
                                                 v1=[v1_1[ang], i[0][0][ang]],
                                                 v2=[v1_2[ang], i[0][1][ang]])
        mutual_information = inv_prop.m_iajb
        mutual_information = np.diag(mutual_information).sum()
        data.append([ang*10, mutual_information, f'Csp3-F2pz_{i[0][2]}', "Entropía"])

df = pd.DataFrame(data, columns=['angulo', 'Entanglement', 'Virtuals',  'Mutual Information'])
fig = px.line(df, x="angulo", y="Entanglement", animation_frame='Virtuals', color='Mutual Information',
       title="Entropy iajb in the Difluorethane (opt-HF) using combinations of 2 virtuals as antibondings, and base cc-pvdz",
      )
fig.update_layout(    yaxis_title=r'Entrelazamiento' )
fig.show()
fig.write_html("m_iajb_singlet_difluorethane_2exc_2.html", include_mathjax='cdn')