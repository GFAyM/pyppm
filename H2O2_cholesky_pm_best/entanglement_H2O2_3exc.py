import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append("/home/fer/pyPPE/src")
from ppe import inverse_principal_propagator
from help_functions import extra_functions
from cloppa import Cloppa
import itertools
import plotly.express as px
import pandas as pd
import itertools


data = []
for ang in range(1,18,1):
    mol, mo_coeff = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
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
    

    V = [(orbital_1[0],orbital_2[0], '1'), (orbital_1[1],orbital_2[1], '2'), (orbital_1[2],orbital_2[2], '3'), 
    (orbital_1[3],orbital_2[3], '4'),(orbital_1_1[0],orbital_2_1[0],'5')]#, (orbital_1[4],orbital_2[4], '5')]
    for i,j,k in itertools.combinations(V,3):
        inv_prop = inverse_principal_propagator(mol=mol, mo_coeff=mo_coeff, 
                                            o1=occ_1, o2=occ_2, 
                                            v1=[i[0],j[0],k[0]], v2=[i[1],j[1],k[1]])
        mutual_information = inv_prop.mutual_information
        data.append([ang*10, mutual_information, f'{i[2]}_{j[2]}_{k[2]}', "Entropía"])


df = pd.DataFrame(data, columns=['angulo',  'Entanglement', 'Virtuals',  'Mutual Information'])
fig = px.line(df, x="angulo", y="Entanglement",  animation_frame='Virtuals', color='Mutual Information',
       title="Mutual Information in the Difluorethane using combinations of 3 virtuals as antibondings, and base cc-pvdz",
      )
fig.update_layout(    yaxis_title=r'Entrelazamiento' )

fig.write_html("mutual_inf_h2o2_3exc.html", include_mathjax='cdn')
#