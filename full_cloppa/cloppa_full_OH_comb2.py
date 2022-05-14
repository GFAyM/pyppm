import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa_full
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import ao2mo
import itertools


M_list = []

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(10*ang)

    viridx = np.where(mo_occ_loc==0)[0]
    occidx = np.where(mo_occ_loc==2)[0]


    full_M_obj = Cloppa_full(
        mol_input=mol_H2O2,basis='6-31G**',
        mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, vir=viridx,
        mo_occ_loc=mo_occ_loc)

    m = full_M_obj.M
    p = np.linalg.inv(m)
    p = p.reshape(9,29,9,29)


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
    
    for i,j in itertools.combinations(V, 2):

        oh_pathway = p[occidx_OH1, i[0]-9, occidx_OH2, i[1]-9]        
        oh_pathway += p[occidx_OH1, j[0]-9, occidx_OH2, j[1]-9]        
        oh_pathway += p[occidx_OH1, i[0]-9, occidx_OH2, j[1]-9]        
        oh_pathway += p[occidx_OH1, j[0]-9, occidx_OH2, i[1]-9]        
    

        M_list.append([ang*10, oh_pathway,f'{i[2]}_{j[2]}'])



df = pd.DataFrame(M_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="P_iajb pathway",
      )
fig.update_layout(    yaxis_title=r'M matrix' )

fig.write_html("Princ_prop_OH_pathways_comb2.html", include_mathjax='cdn')
        


