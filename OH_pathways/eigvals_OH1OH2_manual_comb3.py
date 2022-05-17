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

#    print(m)
    for I,J,K in itertools.combinations(V, 3):
            i = mo_coeff_loc[:, [occidx_OH1]]
            a = mo_coeff_loc[:, [I[0],J[0],K[0]]]
            j = mo_coeff_loc[:, [occidx_OH2]]
            b = mo_coeff_loc[:, [I[1],J[1],K[1]]]     

            m_12 = -ao2mo.general(mol_loc, [a,b,j,i], compact=False)
            m_12 = m_12.reshape(len(a[0]),len(a[0]))
            m_12 -= ao2mo.general(mol_loc, [i,b,j,a], compact=False)
        
            #m_21 = -ao2mo.general(mol_loc, [b,a,i,j], compact=False)
            #m_21 = m_21.reshape(len(a[0]),len(a[0]))
            #m_21 -= ao2mo.general(mol_loc, [j,a,i,b], compact=False)
        
            #m_11 = np.zeros((len(a[0]),len(a[0])))
            #m_22 = np.zeros((len(a[0]),len(a[0])))
            #block1 = np.concatenate((m_11, m_12), axis=1)
            #block2 = np.concatenate((m_21, m_22), axis=1)
            #m = np.concatenate((block1, block2), axis=0)

            #sum_eig = np.linalg.eigvals(m).sum()

            M_list.append([ang*10, m_12.sum(),f'{I[2]}_{J[2]}_{K[2]}'])



df = pd.DataFrame(M_list, columns=['angulo', 'M', 'Virtuals'])


fig = px.line(df, x="angulo", y="M", animation_frame='Virtuals', 
       title="P_iajb pathway",
      )
fig.show()
fig.update_layout(    yaxis_title=r'M matrix' )

fig.write_html("eigvals_OH1OH2_manual_comb3.html", include_mathjax='cdn')
        




