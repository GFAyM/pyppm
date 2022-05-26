from numbers import Real
import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.ppe import inverse_principal_propagator
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import ao2mo
import itertools


M_list = []

for ang in range(1,17,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"wave_H2O2_{ang*10}_new.molden").extraer_coeff

    occidx_OH1 = 7
    viridx_OH1_1s = 10
    viridx_OH1_2s = 32
    viridx_OH1_2px = 34
    viridx_OH1_2py = 33
    viridx_OH1_2pz = 35

    occidx_OH2 = 8
    viridx_OH2_1s = 11
    viridx_OH2_2s = 36
    viridx_OH2_2px = 38
    viridx_OH2_2py = 37
    viridx_OH2_2pz = 39

    V = [(viridx_OH1_1s,viridx_OH2_1s, "1s"), (viridx_OH1_2py, viridx_OH2_2py, "2py"), (viridx_OH1_2px, viridx_OH2_2px, "2px"),
         (viridx_OH1_2pz, viridx_OH2_2pz, "2pz"), (viridx_OH1_2s, viridx_OH2_2s, "2s")]


    for I,J,K in itertools.combinations(V,3):
            i = [I[0],J[0],K[0]]
            j = [I[1],J[1],K[1]]
            m_obj = inverse_principal_propagator(o1=[occidx_OH1], o2=[occidx_OH2], v1=i, v2=j, mo_coeff=mo_coeff_loc, mol=mol_loc)
            m = np.diag(m_obj.m_iajb)
            #sum_eig = np.sum(np.linalg.eigvals(ent))
            M_list.append([ang*10, m.sum() ,f'{I[2]}_{J[2]}_{K[2]}'])



df = pd.DataFrame(M_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="M_iajb pathway",
      )
fig.show()
fig.update_layout(    yaxis_title=r'M matrix' )

fig.write_html("M_OH1OH2_comb3_NLMO.html", include_mathjax='cdn')
        




