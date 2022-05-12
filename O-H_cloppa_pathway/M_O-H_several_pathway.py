import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa_pathways
import plotly.express as px
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


M_diag_list = []


for ang in range(10,11,1):
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


    
    path_OH = Cloppa_pathways(
        mol_input=mol_H2O2,basis='6-31G**',
        mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, 
        o1=[occidx_OH1], o2=[occidx_OH2], 
        v1=[occidx_OH1],
        v2=[occidx_OH2])        

    #m1 = path_OH.M

    #print(m1)
    m2 = path_OH.elements
    print(m2)
    print(m2.shape)
    #print(np.isclose(m1, m2))