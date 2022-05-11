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

    viridx = np.where(mo_occ_loc==0)[0]
    #occidx = np.where(mo_occ_loc==2)[0]
    occidx_OH = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'H', .3, .5)

    occidx_1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('O1 1s', 0.5, 1)
    occidx_2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('O2 1s', 0.5, 1)

    occidx_O_1s = occidx_1 + occidx_2

    occidx_O1_2s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O1 2s', 0.3, 1) #.4 de 2s, 0.6 de 2p
    occidx_O2_2s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O2 2s', 0.3, 1) #.4 de 2s, 0.6 de 2p
    
    occidx_O_2s = occidx_O1_2s + occidx_O2_2s

    occidx_O1_2p = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O1 2p', 0.7, 1) #.2% de 2ps
    occidx_O2_2p = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O2 2p', 0.7, 1) #.2% de 2ps

    occidx_O_2p = occidx_O1_2p + occidx_O2_2p

    occidx_O_O = [extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O1 2px', 0.4, 0.5)[0]]
            


    O = [(occidx_OH,"OH"),(occidx_O_O,"O-O"),(occidx_O_1s,"1sO"),(occidx_O_2s,"2sO"),(occidx_O_2p,"2pO")]
    occidx =  occidx_OH + occidx_O_O + occidx_O_1s + occidx_O_2s + occidx_O_2p


    for i,j,k in itertools.combinations(O,3):
        full_M_obj = Cloppa_full(
            mol_input=mol_H2O2,basis='6-31G**',
            mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, 
            vir=viridx, occ=(i[0]+j[0]+k[0]))#,
            #mo_occ_loc=mo_occ_loc)        
        m = full_M_obj.M
        
        diag_m = np.sum(np.diag(m))
        diag_princ_prop = np.sum(np.diag(np.linalg.inv(m)))
        
        M_diag_list.append([ang*10, diag_m,f'{i[1]}_{j[1]}_{k[1]}'] )
        inv_M_diag_list.append([ang*10, diag_princ_prop,f'{i[1]}_{j[1]}_{k[1]}'])
        

df = pd.DataFrame(M_diag_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="Diag of M matrix using combinations of occupied MO",
      )
fig.update_layout(    yaxis_title=r'M matrix' )

fig.write_html("M_matrix_occ_comb3.html", include_mathjax='cdn')


df = pd.DataFrame(inv_M_diag_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="Principal propagator matrix (diagonal) using combinations of occupied MO",
      )
fig.update_layout(    yaxis_title=r'M matrix' )

fig.write_html("Principal_prop_matrix_occ_comb3.html", include_mathjax='cdn')

