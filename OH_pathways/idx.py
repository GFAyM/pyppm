import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.ppe import inverse_principal_propagator
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import ao2mo
import itertools


M_list = []

for ang in range(1,18,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff

    #viridx_OH2_1s = extra_functions(
    #    molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization(
    #        'H4', .5, .7)


    #viridx_OH1_1s = extra_functions(
    #    molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization(
    #        'H3', .5, .7)

    viridx_OH1_ = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization(
            'O1 3dz', .45, 1)
        
    viridx_OH2_ = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization(
            'O2 3dz', .45, 1)

    print(ang,viridx_OH1_)
    print(ang,viridx_OH2_)    
