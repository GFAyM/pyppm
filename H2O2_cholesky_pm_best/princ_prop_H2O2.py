import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append("/home/fer/pyPPE/src")
from help_functions import extra_functions
from cloppa import Cloppa
import itertools
import plotly.express as px
import pandas as pd
import numpy as np





M_list = []
M_diag_list = []
for ang in range(1, 18, 1):

    mol_loc, mo_coeff_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
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

    mol_H2O2 = '''
        O1   1
        O2   1 1.45643942
        H3   2 0.97055295  1 99.79601616
        H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(ang)
    V1 = [orbital_1[0], orbital_1[1], orbital_1[3], orbital_1_1[0]]    
    V2 = [orbital_2[0], orbital_2[1], orbital_2[3], orbital_2_1[0]]
    cloppa_obj = Cloppa(mol_input=mol_H2O2, basis='6-31G**', mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc,
    o1=occ_1, o2=occ_2, v1=V1, v2=V2)      
    M = cloppa_obj.inverse_prop_pol
    M_list.append([ang*10, np.sum(np.diag(M)),  "Propagador Pol"])
    M_diag_list.append([ang*10, np.sum(np.diag(np.linalg.inv(M))),  "Propagador Pol"])

df = pd.DataFrame(M_list, columns=['angulo', 'Propagator',   'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  color='Polarization Propagator',
       title="Principal propagator of H2O2 using combinations of 4 virtuals as antibondings, and base 6-31G**",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("diag_M.html", include_mathjax='cdn')


df = pd.DataFrame(M_diag_list, columns=['angulo', 'Propagator',   'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  color='Polarization Propagator',
       title="Inverse of principal propagator of H2O2 using combinations of 4 virtuals as antibondings, and base 6-31G**",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("inv_M_diag.html", include_mathjax='cdn')



