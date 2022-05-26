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

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff

    viridx_OH2_1s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H4', .5, .7)

    occidx_OH2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H4', .3, .5)

    viridx_OH1_1s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H3', .5, .7)

    occidx_OH1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H3', .3, .5)

    viridx_OH2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'H4', .7, 1)


    viridx_OH2_2s = viridx_OH2[0]
    viridx_OH2_2pz = viridx_OH2[1]
    viridx_OH2_2py = viridx_OH2[2]
    viridx_OH2_2px = viridx_OH2[3]

    viridx_OH1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'H3', .7, 1)

    viridx_OH1_2s = viridx_OH1[0]
    viridx_OH1_2pz = viridx_OH1[1]
    viridx_OH1_2py = viridx_OH1[2]
    viridx_OH1_2px = viridx_OH1[3]

    V = [(viridx_OH1_1s,viridx_OH2_1s, "1s"), (viridx_OH1_2py, viridx_OH2_2py, "2py"), (viridx_OH1_2px, viridx_OH2_2px, "2px"),
         (viridx_OH1_2pz, viridx_OH2_2pz, "2pz"), (viridx_OH1_2s, viridx_OH2_2s, "2s")]


    for I,J in itertools.combinations(V,2):
            i = [I[0],J[0]]
            j = [I[1],J[1]]
            m_obj = inverse_principal_propagator(o1=[occidx_OH1], o2=[occidx_OH2], v1=i, v2=j, mo_coeff=mo_coeff_loc, mol=mol_loc)
            ent = m_obj.m_iajb
            sum_eig = np.sum(np.linalg.eigvals(ent))
            M_list.append([ang*10, np.real(sum_eig) ,f'{I[2]}_{J[2]}'])



df = pd.DataFrame(M_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="P_iajb pathway",
      )
fig.show()
fig.update_layout(    yaxis_title=r'M matrix' )

#fig.write_html("eigvals_OH1OH2_comb4.html", include_mathjax='cdn')
        




