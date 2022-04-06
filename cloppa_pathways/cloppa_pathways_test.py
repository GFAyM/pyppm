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



M_list = []
M_diag_list = []
inv_M_diag_list = []

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    
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
    
    OCCUPIED = [occ_1[0], occ_2]
    VIRTUAL = [orbital_1[0], orbital_1[1], orbital_1[3], orbital_1_1[0],
               orbital_2[0], orbital_2[1], orbital_2[3], orbital_2_1[0]]

    print(VIRTUAL, type(VIRTUAL))