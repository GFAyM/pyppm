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


    occidx_OH2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H4', .3, .5)

    occidx_OH1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H3', .3, .5)


    viridx_OH2_1s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H4', .5, .7)

    viridx_OH1_1s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H3', .5, .7)


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

    viridx_OH2_O3s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'O1 3s', .45, 1)
        
    viridx_OH1_O3s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'O2 3s', .45, 1)

    viridx_OH2_O3dz = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'O1 3dz', .45, 1)
        
    viridx_OH1_O3dz = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'O2 3dz', .45, 1)
    

    print(f"Angulo: {ang*10}", occidx_OH1, viridx_OH1_2px, viridx_OH1_2pz)
    print(f"Angulo: {ang*10}", occidx_OH2, viridx_OH2_2px, viridx_OH2_2pz) 