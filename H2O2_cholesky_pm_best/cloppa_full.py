import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import full_M_two_elec
import plotly.express as px
import pandas as pd
import numpy as np



M_list = []
M_diag_list = []
inv_M_diag_list = []

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    full_M_obj = full_M_two_elec(mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, mo_occ_loc=mo_occ_loc)
    m = full_M_obj.M
    

    M_list.append([ang*10, np.sum(m),  "Propagador Pol"])
    M_diag_list.append([ang*10, np.sum(np.diag(m)),  "Propagador Pol"])
    inv_M_diag_list.append([ang*10, np.sum(np.diag(np.linalg.inv(m))),  "Propagador Pol"])


df = pd.DataFrame(M_list, columns=['angulo', 'Propagator',   'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  color='Polarization Propagator',
       title="Principal propagator of H2O2 using all the posible excitations",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("M_full_loc.html", include_mathjax='cdn')


df = pd.DataFrame(M_diag_list, columns=['angulo', 'Propagator',   'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  color='Polarization Propagator',
       title="Principal propagator of H2O2 using all the posible excitations",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("M_full_loc_diag.html", include_mathjax='cdn')

df = pd.DataFrame(inv_M_diag_list, columns=['angulo', 'Propagator',   'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  color='Polarization Propagator',
       title="Principal propagator of H2O2 using all the posible excitations",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("inv_M_full_loc_diag.html", include_mathjax='cdn')



