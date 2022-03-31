import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append("/home/bajac/pyPPE/src")
from help_functions import extra_functions
from cloppa import Cloppa_test
import itertools
import plotly.express as px
import pandas as pd
import numpy as np





M_list = []
M_diag_list = []
for ang in range(1, 18, 1):

    mol_loc, mo_coeff_loc, mo_occ = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    occ_1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('H3', .3, .4)
    occ_2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('H4', .3, .4)
    orbital_1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('H3', .7, 1)
    orbital_2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('H4', .7, 1)
    orbital_1_1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('H3', .6, .7)
    orbital_2_1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('H4', .6, .7)

    V = [(orbital_1[0],orbital_2[0], '1'), (orbital_1[1],orbital_2[1], '2'), 
    (orbital_1[2],orbital_2[2], '3'), 
    (orbital_1[3],orbital_2[3], '4'),(orbital_1_1[0],orbital_2_1[0],'5')]#, (orbital_1[4],orbital_2[4], '5')]
    for i,j,k,l in itertools.combinations(V,4):

        cloppa_obj = Cloppa_test(mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc,
        o1=occ_1, o2=occ_2, v1=[i[0],j[0],k[0],l[0]], v2=[i[1],j[1],k[1],l[1]])      
        M = cloppa_obj.inverse_prop_pol
        M_list.append([ang*10, np.sum(np.diag(np.linalg.inv(M))),f'{i[2]}_{j[2]}_{k[2]}_{l[2]}',  "Propagador Pol"])
        M_diag_list.append([ang*10, np.sum(np.diag(M)),f'{i[2]}_{j[2]}_{k[2]}_{l[2]}',  "Propagador Pol"])


df = pd.DataFrame(M_list, columns=['angulo', 'Propagator', 'Virtuals',  'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  animation_frame='Virtuals', color='Polarization Propagator',
       title="Principal propagator of H2O2 using combinations of 4 virtuals as antibondings, and base 6-31G**",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("inv_M_diag.html", include_mathjax='cdn')


df = pd.DataFrame(M_diag_list, columns=['angulo', 'Propagator', 'Virtuals',  'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  animation_frame='Virtuals', color='Polarization Propagator',
       title="Inverse of principal propagator of H2O2 using combinations of 4 virtuals as antibondings, and base 6-31G**",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("M_diag.html", include_mathjax='cdn')



