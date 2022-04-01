import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import full_M_two_elec

from pyscf import gto, scf
import plotly.express as px
import pandas as pd
import numpy as np



M_list = []
M_diag_list = []
inv_M_diag_list = []
for ang in range(1,18,1):
    mol_H2O2 = '''
        O1   1
        O2   1 1.45643942
        H3   2 0.97055295  1 99.79601616
        H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(ang)
    mol = gto.M(atom=str(mol_H2O2), basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    full_M_obj = full_M_two_elec(mo_coeff_loc=mf.mo_coeff, mol_loc=mol, mo_occ_loc=mf.mo_occ)
    m = full_M_obj.M
    print(m.sum())

    M_list.append([ang*10, np.sum(m),  "Propagador Pol"])
    M_diag_list.append([ang*10, np.sum(np.diag(m)),  "Propagador Pol"])
    inv_M_diag_list.append([ang*10, np.sum(np.diag(np.linalg.inv(m))),  "Propagador Pol"])


df = pd.DataFrame(M_list, columns=['angulo', 'Propagator',   'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  color='Polarization Propagator',
       title="Principal propagator of H2O2 using all the posible excitations",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("M_2e_canonical.html", include_mathjax='cdn')


df = pd.DataFrame(M_diag_list, columns=['angulo', 'Propagator',   'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  color='Polarization Propagator',
       title="Principal propagator of H2O2 using all the posible excitations",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("M_2e_canonical_diag.html", include_mathjax='cdn')

df = pd.DataFrame(inv_M_diag_list, columns=['angulo', 'Propagator',   'Polarization Propagator'])
fig = px.line(df, x="angulo", y="Propagator",  color='Polarization Propagator',
       title="Principal propagator of H2O2 using all the posible excitations",
      )
fig.update_layout(    yaxis_title=r'Propagator' )

fig.write_html("inv_M_2e_canonical_diag.html", include_mathjax='cdn')



