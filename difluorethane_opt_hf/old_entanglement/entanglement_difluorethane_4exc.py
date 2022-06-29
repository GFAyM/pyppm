#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.append("/home/fer/pyPPE/src")

from ppe import inverse_principal_propagator 

#from src.ppe import inverse_principal_propagator
from help_functions import extra_functions
import itertools
import plotly.express as px
import pandas as pd


o1 = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 9, 9, 9, 9]
o2 = [10, 10, 8, 8, 8, 8, 8, 8, 8, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8]


v1_1 = [73, 74, 74, 73, 73, 73, 73, 73, 74, 74, 74, 73, 73, 73, 73, 73, 73, 73, 73]
v1_2 = [74, 73, 73, 74, 75, 74, 75, 75, 73, 73, 73, 75, 75, 75, 75, 75, 75, 75, 74]

v2_1 = [22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]
v2_2 = [23, 23, 22, 25, 25, 25, 22, 22, 22, 22, 22, 22, 23, 23, 25, 22, 22, 22, 22]

v3_1 = [41, 40, 39, 40, 40, 40, 39, 40, 40, 41, 40, 40, 39, 40, 40, 40, 40, 41, 41]
v3_2 = [40, 41, 40, 41, 45, 41, 43, 41, 39, 40, 39, 41, 43, 45, 45, 43, 41, 40, 40]

v4_1 = [63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 63]
v4_2 = [64, 64, 64, 65, 62, 62, 69, 69, 69, 69, 69, 69, 69, 69, 62, 65, 64, 63, 64]

       #0   1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18 
v5_1= [46, 45, 45, 45, 46, 45, 46, 46, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46]
v5_2= [45, 51, 50, 50, 41, 46, 47, 48, 51, 46, 51, 48, 47, 41, 41, 42, 48, 45, 45]

v6_1= [49, 50, 51, 51, 50, 51, 51, 50, 50, 50, 50, 49, 49, 50, 50, 51, 50, 51, 50]
v6_2= [50, 48, 49, 48, 48, 49, 50, 49, 49, 49, 49, 50, 51, 51, 48, 49, 49, 50, 49]


V = [(v1_1,v1_2,"Csp3-F2pz"),(v2_1,v2_2,"F3pz"),(v3_1,v3_2,"F3s"),(v4_1,v4_2,"F3dz"), (v5_1,v5_2,"F3py"),(v6_1,v6_2,"F3px")]


data = []
for ang in range(0,19,1):
    for i,j,k,l in itertools.combinations(V,4):
        mol, mo_coeff = extra_functions(molden_file=f"difluorethane_{ang*10}_Cholesky_PM.molden").extraer_coeff
        inv_prop = inverse_principal_propagator(mol=mol, mo_coeff=mo_coeff, o1=[o1[ang]], o2=[o2[ang]], 
                                                v1=[i[0][ang], j[0][ang], k[0][ang], l[0][ang]],
                                                v2=[i[1][ang], j[1][ang], k[1][ang], l[1][ang]])
        mutual_information = inv_prop.mutual_information
        data.append([ang*10, mutual_information, f'{i[2]}_{j[2]}_{k[2]}_{l[2]}', "Entropía"])

df = pd.DataFrame(data, columns=['angulo', 'Entanglement', 'Virtuals',  'Mutual Information'])
fig = px.line(df, x="angulo", y="Entanglement", animation_frame='Virtuals', color='Mutual Information',
       title="Mutual Information Triplet in the Difluorethane using combinations of 4 virtuals as antibondings, base cc-pvdz",
      )
fig.update_layout(    yaxis_title=r'Entrelazamiento' )

fig.write_html("mutual_inf_triplet_difluorethane_4exc.html", include_mathjax='cdn')
#

data = []
for ang in range(0,19,1):
    for i,j,k,l in itertools.combinations(V,4):
        mol, mo_coeff = extra_functions(molden_file=f"difluorethane_{ang*10}_Cholesky_PM.molden").extraer_coeff
        inv_prop = inverse_principal_propagator(mol=mol, mo_coeff=mo_coeff, spin_dependence="singlet", 
                                            o1=[o1[ang]], o2=[o2[ang]], 
                                            v1=[i[0][ang], j[0][ang], k[0][ang], l[0][ang]],
                                            v2=[i[1][ang], j[1][ang], k[1][ang], l[1][ang]])
        mutual_information = inv_prop.mutual_information
        data.append([ang*10, mutual_information, f'{i[2]}_{j[2]}_{k[2]}_{l[2]}', "Entropía"])

df = pd.DataFrame(data, columns=['angulo', 'Entanglement', 'Virtuals',  'Mutual Information'])
fig = px.line(df, x="angulo", y="Entanglement", animation_frame='Virtuals', color='Mutual Information',
       title="Mutual Information Singlet in the Difluorethane using combinations of 4 virtuals as antibondings, base cc-pvdz",
      )
fig.update_layout(    yaxis_title=r'Entrelazamiento' )

fig.write_html("mutual_inf_singlet_difluorethane_4exc.html", include_mathjax='cdn')

