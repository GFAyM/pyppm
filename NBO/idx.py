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
from pyscf.tools import mo_mapping


M_list = []

for ang in range(13,15,1):

    viridx_OH2_1s = extra_functions(
        molden_file=f"wave_H2O2_{ang*10}_new.molden").mo_hibridization(
            'O1 3dzz', 0.1, 1, cart=True, orth_method='lowdin')
    print(ang*10,viridx_OH2_1s)

    viridx_OH1_1s = extra_functions(
        molden_file=f"wave_H2O2_{ang*10}_new.molden").mo_hibridization(
            'O2 3dzz', 0.1, 1, cart=True, orth_method='lowdin')
    
    print(ang*10,viridx_OH1_1s)


    #print(ang,viridx_OH2_1s)
    #viridx_OH1_1s = extra_functions(
    #    molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization(
    #        'H3', .5, .7)

