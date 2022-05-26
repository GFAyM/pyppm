from numbers import Real
import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa_full
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
    #mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(10*ang)

    viridx = np.where(mo_occ_loc==0)[0]
    occidx = np.where(mo_occ_loc==2)[0]

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

#    viridx_O1_3dz = 17
#    viridx_O2_3dz = 27
    
#    viridx_O1_3dzz = 15
#    viridx_O2_3dzz = 25


    V = [(viridx_OH1_1s,viridx_OH2_1s, "1s"), (viridx_OH1_2py, viridx_OH2_2py, "2py"), (viridx_OH1_2px, viridx_OH2_2px, "2px"),
         (viridx_OH1_2pz, viridx_OH2_2pz, "2pz"), (viridx_OH1_2s, viridx_OH2_2s, "2s")]#, (viridx_O1_3dz, viridx_O2_3dz, "3dz"),
         #(viridx_O1_3dzz, viridx_O2_3dzz, "3dzz")]


    full_M_obj = Cloppa_full(
        mol_input=mol_H2O2,basis='6-31G**',
        mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, vir=viridx,
        mo_occ_loc=mo_occ_loc)
    
    for i,j in itertools.combinations(V, 2):
        m = 0
        m += full_M_obj.elements_p(occidx_OH1, i[0], occidx_OH2, i[1])
        m += full_M_obj.elements_p(occidx_OH1, j[0], occidx_OH2, j[1])

#        print(m)
#        oh_pathway = np.sum(p[occidx_OH1, [i[0]-9,j[0]-9,k[0]-9,l[0]-9], 
#                        occidx_OH2, [i[1]-9,j[1]-9,k[1]-9,l[1]-9]])        
        
        M_list.append([ang*10, m,f'{i[2]}_{j[2]}'])



df = pd.DataFrame(M_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="P_iajb pathway",
      )
fig.update_layout(    yaxis_title=r'M matrix' )
fig.show()
fig.write_html("Princ_prop_OH1OH2_comb2_NLMO.html", include_mathjax='cdn')



