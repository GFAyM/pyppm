import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa_full
import plotly.express as px
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


M_diag_list = []
inv_M_diag_list = []

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(10*ang)

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
            
    viridx_OH2_1s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H4', .5, 0.7)

    viridx_OH1_1s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H3', .5, 0.7)



    for i,j,k,l,n in itertools.combinations(O,5):
        full_M_obj = Cloppa_full(
            mol_input=mol_H2O2,basis='6-31G**',
            mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, 
            vir=viridx, occ=(i[0]+j[0]+k[0]+l[0]+n[0]))#,
            #mo_occ_loc=mo_occ_loc)        
        m = full_M_obj.M
        
        diag_m = np.sum(np.diag(m))
        diag_princ_prop = np.sum(np.diag(np.linalg.inv(m)))
        
        M_diag_list.append([ang*10, diag_m,f'{i[1]}_{j[1]}_{k[1]}_{l[1]}_{n[1]}'] )
        inv_M_diag_list.append([ang*10, diag_princ_prop,f'{i[1]}_{j[1]}_{k[1]}_{l[1]}_{n[1]}'])
        

df = pd.DataFrame(M_diag_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="Diag of M matrix using combinations of occupied MO",
      )
fig.update_layout(    yaxis_title=r'M matrix' )

fig.write_html("M_matrix_occ_comb5.html", include_mathjax='cdn')


df = pd.DataFrame(inv_M_diag_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="Principal propagator matrix (diagonal) using combinations of occupied MO",
      )
fig.update_layout(    yaxis_title=r'M matrix' )

fig.write_html("Principal_prop_matrix_occ_comb5.html", include_mathjax='cdn')

